## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000229875


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0129602, -0.0112432, -0.0129602, -0.0112432, -0.0010847, 0.0010847)
1: (-0.0065926, -0.0061085, -0.0065926, -0.0061085, -0.0003058, 0.0003058)
2: (-0.0100820, -0.0065102, -0.0100820, -0.0065102, -0.0022564, 0.0022564)
3: (0.0002931, 0.0007658, 0.0002931, 0.0007658, -0.0002986, 0.0002986)
4: (0.0109572, 0.0136265, 0.0109572, 0.0136265, -0.0016863, 0.0016863)
5: (0.9985505, 0.9992921, 0.9985505, 0.9992921, -0.0004685, 0.0004685)
6: (0.0065679, 0.0072411, 0.0065679, 0.0072411, -0.0004253, 0.0004253)
7: (0.0011288, 0.0036409, 0.0011288, 0.0036409, -0.0015870, 0.0015870)
8: (-0.0120266, -0.0100714, -0.0120266, -0.0100714, -0.0012352, 0.0012352)
9: (-0.0031408, -0.0029721, -0.0031408, -0.0029721, -0.0001066, 0.0001066)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 1.41 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0002807, upper bound: 0.0002809

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002653, upper bound: 0.0002505
time: 0.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002652, upper bound: 0.0002653
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 5, lower bound: -0.0002653, upper bound: 0.0002505
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 5, lower bound: -0.0002652, upper bound: 0.0002653

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0128153, -0.0112545, -0.0129178, -0.0112436, -0.0008977, 0.0009608
1: -0.0065518, -0.0061117, -0.0065807, -0.0061086, -0.0002531, 0.0002709
2: -0.0097806, -0.0065338, -0.0099939, -0.0065110, -0.0018674, 0.0019986
3: 0.0003330, 0.0007626, 0.0003048, 0.0007657, -0.0002471, 0.0002645
4: 0.0109748, 0.0134013, 0.0109578, 0.0135607, -0.0014936, 0.0013956
5: 0.9985554, 0.9992295, 0.9985507, 0.9992738, -0.0004150, 0.0003877
6: 0.0065724, 0.0071843, 0.0065681, 0.0072245, -0.0003767, 0.0003520
7: 0.0011454, 0.0034290, 0.0011294, 0.0035790, -0.0014057, 0.0013134
8: -0.0118616, -0.0100843, -0.0119784, -0.0100719, -0.0010222, 0.0010940
9: -0.0031397, -0.0029864, -0.0031408, -0.0029763, -0.0000944, 0.0000882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002506
time: 0.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002506
time: 0.58 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0129202, -0.0112438, -0.0129566, -0.0112432, -0.0008677, 0.0010830
1: -0.0065813, -0.0061087, -0.0065916, -0.0061085, -0.0002446, 0.0003053
2: -0.0099988, -0.0065115, -0.0100745, -0.0065103, -0.0018050, 0.0022529
3: 0.0003041, 0.0007656, 0.0002941, 0.0007658, -0.0002389, 0.0002981
4: 0.0109582, 0.0135643, 0.0109573, 0.0136209, -0.0016837, 0.0013490
5: 0.9985508, 0.9992747, 0.9985505, 0.9992905, -0.0004678, 0.0003748
6: 0.0065682, 0.0072254, 0.0065679, 0.0072397, -0.0004246, 0.0003402
7: 0.0011297, 0.0035824, 0.0011289, 0.0036357, -0.0015845, 0.0012695
8: -0.0119811, -0.0100721, -0.0120225, -0.0100715, -0.0009881, 0.0012333
9: -0.0031408, -0.0029761, -0.0031408, -0.0029725, -0.0001064, 0.0000852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002654
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002653
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002506
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002506
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002654
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 5, lower bound: -0.0002505, upper bound: 0.0002653

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128153, -0.0112545, -0.0128153, -0.0112545, -0.0008310, 0.0008310
1: -0.0065518, -0.0061117, -0.0065518, -0.0061117, -0.0002343, 0.0002343
2: -0.0097806, -0.0065338, -0.0097806, -0.0065338, -0.0017287, 0.0017287
3: 0.0003330, 0.0007626, 0.0003330, 0.0007626, -0.0002288, 0.0002288
4: 0.0109748, 0.0134013, 0.0109748, 0.0134013, -0.0012919, 0.0012919
5: 0.9985554, 0.9992295, 0.9985554, 0.9992295, -0.0003589, 0.0003589
6: 0.0065724, 0.0071843, 0.0065724, 0.0071843, -0.0003258, 0.0003258
7: 0.0011454, 0.0034290, 0.0011454, 0.0034290, -0.0012159, 0.0012159
8: -0.0118616, -0.0100843, -0.0118616, -0.0100843, -0.0009463, 0.0009463
9: -0.0031397, -0.0029864, -0.0031397, -0.0029864, -0.0000816, 0.0000816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002455, upper bound: 0.0002411
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002454
time: 0.59 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128153, -0.0112545, -0.0129202, -0.0112438, -0.0008975, 0.0010059
1: -0.0065518, -0.0061117, -0.0065813, -0.0061087, -0.0002530, 0.0002836
2: -0.0097806, -0.0065338, -0.0099988, -0.0065115, -0.0018670, 0.0020925
3: 0.0003330, 0.0007626, 0.0003041, 0.0007656, -0.0002471, 0.0002769
4: 0.0109748, 0.0134013, 0.0109582, 0.0135643, -0.0015638, 0.0013953
5: 0.9985554, 0.9992295, 0.9985508, 0.9992747, -0.0004345, 0.0003877
6: 0.0065724, 0.0071843, 0.0065682, 0.0072254, -0.0003944, 0.0003519
7: 0.0011454, 0.0034290, 0.0011297, 0.0035824, -0.0014717, 0.0013131
8: -0.0118616, -0.0100843, -0.0119811, -0.0100721, -0.0010220, 0.0011454
9: -0.0031397, -0.0029864, -0.0031408, -0.0029761, -0.0000988, 0.0000882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002455, upper bound: 0.0002410
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002455
time: 0.58 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0129202, -0.0112438, -0.0128153, -0.0112545, -0.0010059, 0.0008975
1: -0.0065813, -0.0061087, -0.0065518, -0.0061117, -0.0002836, 0.0002530
2: -0.0099988, -0.0065115, -0.0097806, -0.0065338, -0.0020925, 0.0018670
3: 0.0003041, 0.0007656, 0.0003330, 0.0007626, -0.0002769, 0.0002471
4: 0.0109582, 0.0135643, 0.0109748, 0.0134013, -0.0013953, 0.0015638
5: 0.9985508, 0.9992747, 0.9985554, 0.9992295, -0.0003877, 0.0004345
6: 0.0065682, 0.0072254, 0.0065724, 0.0071843, -0.0003519, 0.0003944
7: 0.0011297, 0.0035824, 0.0011454, 0.0034290, -0.0013131, 0.0014717
8: -0.0119811, -0.0100721, -0.0118616, -0.0100843, -0.0011454, 0.0010220
9: -0.0031408, -0.0029761, -0.0031397, -0.0029864, -0.0000882, 0.0000988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002560
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002603
time: 0.62 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0129202, -0.0112438, -0.0129202, -0.0112438, -0.0008672, 0.0008672
1: -0.0065813, -0.0061087, -0.0065813, -0.0061087, -0.0002445, 0.0002445
2: -0.0099988, -0.0065115, -0.0099988, -0.0065115, -0.0018040, 0.0018040
3: 0.0003041, 0.0007656, 0.0003041, 0.0007656, -0.0002387, 0.0002387
4: 0.0109582, 0.0135643, 0.0109582, 0.0135643, -0.0013482, 0.0013482
5: 0.9985508, 0.9992747, 0.9985508, 0.9992747, -0.0003746, 0.0003746
6: 0.0065682, 0.0072254, 0.0065682, 0.0072254, -0.0003400, 0.0003400
7: 0.0011297, 0.0035824, 0.0011297, 0.0035824, -0.0012688, 0.0012688
8: -0.0119811, -0.0100721, -0.0119811, -0.0100721, -0.0009875, 0.0009875
9: -0.0031408, -0.0029761, -0.0031408, -0.0029761, -0.0000852, 0.0000852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002560
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002603
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002455, upper bound: 0.0002411
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002454
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002455, upper bound: 0.0002410
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002455
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002560
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002603
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002560
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 5, lower bound: -0.0002453, upper bound: 0.0002603

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127790, -0.0112439, -0.0128045, -0.0112546, -0.0007885, 0.0008184
1: -0.0065415, -0.0061087, -0.0065487, -0.0061117, -0.0002223, 0.0002307
2: -0.0097051, -0.0065118, -0.0097581, -0.0065340, -0.0016403, 0.0017024
3: 0.0003430, 0.0007656, 0.0003360, 0.0007626, -0.0002171, 0.0002253
4: 0.0109584, 0.0133448, 0.0109749, 0.0133844, -0.0012723, 0.0012259
5: 0.9985508, 0.9992138, 0.9985554, 0.9992248, -0.0003535, 0.0003406
6: 0.0065682, 0.0071700, 0.0065724, 0.0071800, -0.0003208, 0.0003091
7: 0.0011299, 0.0033758, 0.0011455, 0.0034131, -0.0011973, 0.0011537
8: -0.0118203, -0.0100723, -0.0118493, -0.0100844, -0.0008979, 0.0009319
9: -0.0031407, -0.0029899, -0.0031397, -0.0029874, -0.0000804, 0.0000775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002374, upper bound: 0.0002352
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002400, upper bound: 0.0002360
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128004, -0.0112547, -0.0128142, -0.0112545, -0.0007991, 0.0008302
1: -0.0065476, -0.0061118, -0.0065515, -0.0061117, -0.0002253, 0.0002341
2: -0.0097497, -0.0065342, -0.0097784, -0.0065338, -0.0016624, 0.0017269
3: 0.0003371, 0.0007626, 0.0003333, 0.0007626, -0.0002200, 0.0002285
4: 0.0109751, 0.0133782, 0.0109749, 0.0133996, -0.0012906, 0.0012424
5: 0.9985554, 0.9992231, 0.9985554, 0.9992290, -0.0003586, 0.0003452
6: 0.0065724, 0.0071784, 0.0065724, 0.0071839, -0.0003255, 0.0003133
7: 0.0011456, 0.0034072, 0.0011454, 0.0034274, -0.0012146, 0.0011692
8: -0.0118447, -0.0100845, -0.0118604, -0.0100843, -0.0009100, 0.0009453
9: -0.0031397, -0.0029878, -0.0031397, -0.0029865, -0.0000816, 0.0000785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002418, upper bound: 0.0002455
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002418, upper bound: 0.0002456
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127790, -0.0112439, -0.0129094, -0.0112439, -0.0008550, 0.0009945
1: -0.0065415, -0.0061087, -0.0065783, -0.0061087, -0.0002411, 0.0002804
2: -0.0097051, -0.0065118, -0.0099763, -0.0065117, -0.0017786, 0.0020687
3: 0.0003430, 0.0007656, 0.0003071, 0.0007656, -0.0002354, 0.0002738
4: 0.0109584, 0.0133448, 0.0109583, 0.0135475, -0.0015460, 0.0013292
5: 0.9985508, 0.9992138, 0.9985508, 0.9992702, -0.0004295, 0.0003693
6: 0.0065682, 0.0071700, 0.0065682, 0.0072212, -0.0003899, 0.0003352
7: 0.0011299, 0.0033758, 0.0011298, 0.0035666, -0.0014550, 0.0012509
8: -0.0118203, -0.0100723, -0.0119688, -0.0100722, -0.0009736, 0.0011324
9: -0.0031407, -0.0029899, -0.0031408, -0.0029771, -0.0000977, 0.0000840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002500, upper bound: 0.0002344
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002551, upper bound: 0.0002352
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128004, -0.0112547, -0.0129192, -0.0112438, -0.0008700, 0.0010049
1: -0.0065476, -0.0061118, -0.0065811, -0.0061087, -0.0002453, 0.0002833
2: -0.0097497, -0.0065342, -0.0099967, -0.0065115, -0.0018097, 0.0020903
3: 0.0003371, 0.0007626, 0.0003044, 0.0007656, -0.0002395, 0.0002766
4: 0.0109751, 0.0133782, 0.0109582, 0.0135627, -0.0015622, 0.0013524
5: 0.9985554, 0.9992231, 0.9985508, 0.9992743, -0.0004340, 0.0003757
6: 0.0065724, 0.0071784, 0.0065682, 0.0072250, -0.0003940, 0.0003411
7: 0.0011456, 0.0034072, 0.0011297, 0.0035809, -0.0014702, 0.0012728
8: -0.0118447, -0.0100845, -0.0119799, -0.0100721, -0.0009906, 0.0011442
9: -0.0031397, -0.0029878, -0.0031408, -0.0029762, -0.0000987, 0.0000855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002560, upper bound: 0.0002454
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002560, upper bound: 0.0002455
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128833, -0.0112362, -0.0128045, -0.0112546, -0.0009669, 0.0008891
1: -0.0065709, -0.0061066, -0.0065487, -0.0061117, -0.0002726, 0.0002507
2: -0.0099219, -0.0064958, -0.0097581, -0.0065340, -0.0020113, 0.0018495
3: 0.0003143, 0.0007677, 0.0003360, 0.0007626, -0.0002662, 0.0002447
4: 0.0109464, 0.0135069, 0.0109749, 0.0133844, -0.0013822, 0.0015031
5: 0.9985474, 0.9992589, 0.9985554, 0.9992248, -0.0003840, 0.0004176
6: 0.0065652, 0.0072109, 0.0065724, 0.0071800, -0.0003486, 0.0003791
7: 0.0011186, 0.0035284, 0.0011455, 0.0034131, -0.0013008, 0.0014146
8: -0.0119390, -0.0100635, -0.0118493, -0.0100844, -0.0011010, 0.0010124
9: -0.0031415, -0.0029797, -0.0031397, -0.0029874, -0.0000873, 0.0000950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002371, upper bound: 0.0002502
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002507
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0129058, -0.0112439, -0.0128142, -0.0112545, -0.0009762, 0.0008966
1: -0.0065773, -0.0061087, -0.0065515, -0.0061117, -0.0002752, 0.0002528
2: -0.0099689, -0.0065118, -0.0097784, -0.0065338, -0.0020308, 0.0018652
3: 0.0003081, 0.0007656, 0.0003333, 0.0007626, -0.0002687, 0.0002468
4: 0.0109584, 0.0135420, 0.0109749, 0.0133996, -0.0013939, 0.0015177
5: 0.9985508, 0.9992685, 0.9985554, 0.9992290, -0.0003873, 0.0004217
6: 0.0065682, 0.0072198, 0.0065724, 0.0071839, -0.0003515, 0.0003827
7: 0.0011299, 0.0035614, 0.0011454, 0.0034274, -0.0013118, 0.0014283
8: -0.0119647, -0.0100723, -0.0118604, -0.0100843, -0.0011116, 0.0010210
9: -0.0031407, -0.0029775, -0.0031397, -0.0029865, -0.0000881, 0.0000959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002409, upper bound: 0.0002603
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002409, upper bound: 0.0002603
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128833, -0.0112362, -0.0129094, -0.0112439, -0.0008249, 0.0008550
1: -0.0065709, -0.0061066, -0.0065783, -0.0061087, -0.0002326, 0.0002410
2: -0.0099219, -0.0064958, -0.0099763, -0.0065117, -0.0017159, 0.0017785
3: 0.0003143, 0.0007677, 0.0003071, 0.0007656, -0.0002271, 0.0002354
4: 0.0109464, 0.0135069, 0.0109583, 0.0135475, -0.0013291, 0.0012823
5: 0.9985474, 0.9992589, 0.9985508, 0.9992702, -0.0003693, 0.0003563
6: 0.0065652, 0.0072109, 0.0065682, 0.0072212, -0.0003352, 0.0003234
7: 0.0011186, 0.0035284, 0.0011298, 0.0035666, -0.0012509, 0.0012068
8: -0.0119390, -0.0100635, -0.0119688, -0.0100722, -0.0009393, 0.0009735
9: -0.0031415, -0.0029797, -0.0031408, -0.0029771, -0.0000840, 0.0000810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002372, upper bound: 0.0002501
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002397, upper bound: 0.0002507
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0129058, -0.0112439, -0.0129192, -0.0112438, -0.0008341, 0.0008663
1: -0.0065773, -0.0061087, -0.0065811, -0.0061087, -0.0002352, 0.0002442
2: -0.0099689, -0.0065118, -0.0099967, -0.0065115, -0.0017351, 0.0018021
3: 0.0003081, 0.0007656, 0.0003044, 0.0007656, -0.0002296, 0.0002385
4: 0.0109584, 0.0135420, 0.0109582, 0.0135627, -0.0013468, 0.0012967
5: 0.9985508, 0.9992685, 0.9985508, 0.9992743, -0.0003742, 0.0003603
6: 0.0065682, 0.0072198, 0.0065682, 0.0072250, -0.0003396, 0.0003270
7: 0.0011299, 0.0035614, 0.0011297, 0.0035809, -0.0012675, 0.0012203
8: -0.0119647, -0.0100723, -0.0119799, -0.0100721, -0.0009498, 0.0009865
9: -0.0031407, -0.0029775, -0.0031408, -0.0029762, -0.0000851, 0.0000819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002410, upper bound: 0.0002602
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002410, upper bound: 0.0002602
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.93 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002374, upper bound: 0.0002352
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002400, upper bound: 0.0002360
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002418, upper bound: 0.0002455
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002418, upper bound: 0.0002456
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002500, upper bound: 0.0002344
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002551, upper bound: 0.0002352
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002560, upper bound: 0.0002454
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002560, upper bound: 0.0002455
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002371, upper bound: 0.0002502
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002396, upper bound: 0.0002507
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002409, upper bound: 0.0002603
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002409, upper bound: 0.0002603
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002372, upper bound: 0.0002501
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002397, upper bound: 0.0002507
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002410, upper bound: 0.0002602
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 5, lower bound: -0.0002410, upper bound: 0.0002602

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127687, -0.0112445, -0.0127720, -0.0112376, -0.0007792, 0.0007779
1: -0.0065386, -0.0061089, -0.0065396, -0.0061070, -0.0002197, 0.0002193
2: -0.0096836, -0.0065129, -0.0096905, -0.0064987, -0.0016209, 0.0016183
3: 0.0003458, 0.0007654, 0.0003449, 0.0007673, -0.0002145, 0.0002142
4: 0.0109592, 0.0133288, 0.0109486, 0.0133339, -0.0012094, 0.0012113
5: 0.9985511, 0.9992094, 0.9985481, 0.9992108, -0.0003360, 0.0003365
6: 0.0065684, 0.0071660, 0.0065657, 0.0071673, -0.0003050, 0.0003055
7: 0.0011307, 0.0033608, 0.0011207, 0.0033656, -0.0011382, 0.0011400
8: -0.0118085, -0.0100729, -0.0118123, -0.0100651, -0.0008873, 0.0008858
9: -0.0031407, -0.0029910, -0.0031414, -0.0029906, -0.0000764, 0.0000765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002198, upper bound: 0.0001971
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002305
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127790, -0.0112439, -0.0127932, -0.0112552, -0.0007880, 0.0007843
1: -0.0065415, -0.0061087, -0.0065455, -0.0061119, -0.0002222, 0.0002211
2: -0.0097051, -0.0065118, -0.0097346, -0.0065352, -0.0016392, 0.0016316
3: 0.0003430, 0.0007656, 0.0003391, 0.0007625, -0.0002169, 0.0002159
4: 0.0109584, 0.0133448, 0.0109759, 0.0133669, -0.0012194, 0.0012250
5: 0.9985508, 0.9992138, 0.9985557, 0.9992199, -0.0003388, 0.0003403
6: 0.0065682, 0.0071700, 0.0065726, 0.0071756, -0.0003075, 0.0003089
7: 0.0011299, 0.0033758, 0.0011464, 0.0033966, -0.0011475, 0.0011529
8: -0.0118203, -0.0100723, -0.0118364, -0.0100851, -0.0008973, 0.0008931
9: -0.0031407, -0.0029899, -0.0031396, -0.0029885, -0.0000771, 0.0000774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002394, upper bound: 0.0002329
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002394, upper bound: 0.0002360
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128004, -0.0112547, -0.0127790, -0.0112439, -0.0008217, 0.0007885
1: -0.0065476, -0.0061118, -0.0065415, -0.0061087, -0.0002317, 0.0002223
2: -0.0097497, -0.0065342, -0.0097051, -0.0065118, -0.0017094, 0.0016401
3: 0.0003371, 0.0007626, 0.0003430, 0.0007656, -0.0002262, 0.0002170
4: 0.0109751, 0.0133782, 0.0109584, 0.0133448, -0.0012257, 0.0012775
5: 0.9985554, 0.9992231, 0.9985508, 0.9992138, -0.0003405, 0.0003549
6: 0.0065724, 0.0071784, 0.0065682, 0.0071700, -0.0003091, 0.0003222
7: 0.0011456, 0.0034072, 0.0011299, 0.0033758, -0.0011536, 0.0012022
8: -0.0118447, -0.0100845, -0.0118203, -0.0100723, -0.0009357, 0.0008978
9: -0.0031397, -0.0029878, -0.0031407, -0.0029899, -0.0000775, 0.0000807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002374
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002360, upper bound: 0.0002400
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128004, -0.0112547, -0.0128004, -0.0112547, -0.0007990, 0.0007990
1: -0.0065476, -0.0061118, -0.0065476, -0.0061118, -0.0002253, 0.0002253
2: -0.0097497, -0.0065342, -0.0097497, -0.0065342, -0.0016621, 0.0016621
3: 0.0003371, 0.0007626, 0.0003371, 0.0007626, -0.0002200, 0.0002200
4: 0.0109751, 0.0133782, 0.0109751, 0.0133782, -0.0012422, 0.0012422
5: 0.9985554, 0.9992231, 0.9985554, 0.9992231, -0.0003451, 0.0003451
6: 0.0065724, 0.0071784, 0.0065724, 0.0071784, -0.0003133, 0.0003133
7: 0.0011456, 0.0034072, 0.0011456, 0.0034072, -0.0011690, 0.0011690
8: -0.0118447, -0.0100845, -0.0118447, -0.0100845, -0.0009099, 0.0009099
9: -0.0031397, -0.0029878, -0.0031397, -0.0029878, -0.0000785, 0.0000785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002375
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002360, upper bound: 0.0002399
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127687, -0.0112445, -0.0128742, -0.0112332, -0.0008480, 0.0009529
1: -0.0065386, -0.0061089, -0.0065684, -0.0061057, -0.0002391, 0.0002686
2: -0.0096836, -0.0065129, -0.0099032, -0.0064894, -0.0017639, 0.0019821
3: 0.0003458, 0.0007654, 0.0003168, 0.0007685, -0.0002334, 0.0002623
4: 0.0109592, 0.0133288, 0.0109416, 0.0134929, -0.0014813, 0.0013183
5: 0.9985511, 0.9992094, 0.9985461, 0.9992550, -0.0004116, 0.0003663
6: 0.0065684, 0.0071660, 0.0065640, 0.0072074, -0.0003736, 0.0003324
7: 0.0011307, 0.0033608, 0.0011142, 0.0035151, -0.0013941, 0.0012406
8: -0.0118085, -0.0100729, -0.0119287, -0.0100600, -0.0009656, 0.0010850
9: -0.0031407, -0.0029910, -0.0031418, -0.0029806, -0.0000936, 0.0000833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002340, upper bound: 0.0001970
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002452, upper bound: 0.0002297
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127790, -0.0112439, -0.0128986, -0.0112444, -0.0008545, 0.0009593
1: -0.0065415, -0.0061087, -0.0065753, -0.0061089, -0.0002409, 0.0002705
2: -0.0097051, -0.0065118, -0.0099539, -0.0065128, -0.0017776, 0.0019955
3: 0.0003430, 0.0007656, 0.0003101, 0.0007654, -0.0002352, 0.0002641
4: 0.0109584, 0.0133448, 0.0109591, 0.0135308, -0.0014913, 0.0013285
5: 0.9985508, 0.9992138, 0.9985510, 0.9992654, -0.0004143, 0.0003691
6: 0.0065682, 0.0071700, 0.0065684, 0.0072169, -0.0003761, 0.0003350
7: 0.0011299, 0.0033758, 0.0011306, 0.0035508, -0.0014035, 0.0012502
8: -0.0118203, -0.0100723, -0.0119565, -0.0100728, -0.0009731, 0.0010924
9: -0.0031407, -0.0029899, -0.0031407, -0.0029782, -0.0000942, 0.0000840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002546, upper bound: 0.0002323
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002546, upper bound: 0.0002352
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128004, -0.0112547, -0.0128833, -0.0112362, -0.0008924, 0.0009668
1: -0.0065476, -0.0061118, -0.0065709, -0.0061066, -0.0002516, 0.0002726
2: -0.0097497, -0.0065342, -0.0099219, -0.0064958, -0.0018564, 0.0020111
3: 0.0003371, 0.0007626, 0.0003143, 0.0007677, -0.0002457, 0.0002661
4: 0.0109751, 0.0133782, 0.0109464, 0.0135069, -0.0015030, 0.0013874
5: 0.9985554, 0.9992231, 0.9985474, 0.9992589, -0.0004176, 0.0003855
6: 0.0065724, 0.0071784, 0.0065652, 0.0072109, -0.0003790, 0.0003499
7: 0.0011456, 0.0034072, 0.0011186, 0.0035284, -0.0014145, 0.0013057
8: -0.0118447, -0.0100845, -0.0119390, -0.0100635, -0.0010162, 0.0011009
9: -0.0031397, -0.0029878, -0.0031415, -0.0029797, -0.0000950, 0.0000877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002372
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002507, upper bound: 0.0002397
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128004, -0.0112547, -0.0129058, -0.0112439, -0.0008698, 0.0009761
1: -0.0065476, -0.0061118, -0.0065773, -0.0061087, -0.0002452, 0.0002752
2: -0.0097497, -0.0065342, -0.0099689, -0.0065118, -0.0018093, 0.0020305
3: 0.0003371, 0.0007626, 0.0003081, 0.0007656, -0.0002394, 0.0002687
4: 0.0109751, 0.0133782, 0.0109584, 0.0135420, -0.0015175, 0.0013522
5: 0.9985554, 0.9992231, 0.9985508, 0.9992685, -0.0004216, 0.0003757
6: 0.0065724, 0.0071784, 0.0065682, 0.0072198, -0.0003827, 0.0003410
7: 0.0011456, 0.0034072, 0.0011299, 0.0035614, -0.0014281, 0.0012726
8: -0.0118447, -0.0100845, -0.0119647, -0.0100723, -0.0009904, 0.0011115
9: -0.0031397, -0.0029878, -0.0031407, -0.0029775, -0.0000959, 0.0000854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002371
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002507, upper bound: 0.0002396
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128729, -0.0112367, -0.0127720, -0.0112376, -0.0009570, 0.0008487
1: -0.0065680, -0.0061067, -0.0065396, -0.0061070, -0.0002698, 0.0002393
2: -0.0099005, -0.0064968, -0.0096905, -0.0064987, -0.0019908, 0.0017654
3: 0.0003171, 0.0007676, 0.0003449, 0.0007673, -0.0002635, 0.0002336
4: 0.0109471, 0.0134909, 0.0109486, 0.0133339, -0.0013194, 0.0014878
5: 0.9985477, 0.9992545, 0.9985481, 0.9992108, -0.0003666, 0.0004134
6: 0.0065654, 0.0072069, 0.0065657, 0.0071673, -0.0003327, 0.0003752
7: 0.0011194, 0.0035133, 0.0011207, 0.0033656, -0.0012417, 0.0014002
8: -0.0119272, -0.0100641, -0.0118123, -0.0100651, -0.0010898, 0.0009664
9: -0.0031415, -0.0029807, -0.0031414, -0.0029906, -0.0000834, 0.0000940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002197, upper bound: 0.0002136
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002450
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128833, -0.0112362, -0.0127932, -0.0112552, -0.0009663, 0.0008575
1: -0.0065709, -0.0061066, -0.0065455, -0.0061119, -0.0002724, 0.0002418
2: -0.0099219, -0.0064958, -0.0097346, -0.0065352, -0.0020101, 0.0017837
3: 0.0003143, 0.0007677, 0.0003391, 0.0007625, -0.0002660, 0.0002360
4: 0.0109464, 0.0135069, 0.0109759, 0.0133669, -0.0013331, 0.0015023
5: 0.9985474, 0.9992589, 0.9985557, 0.9992199, -0.0003704, 0.0004174
6: 0.0065652, 0.0072109, 0.0065726, 0.0071756, -0.0003362, 0.0003788
7: 0.0011186, 0.0035284, 0.0011464, 0.0033966, -0.0012546, 0.0014138
8: -0.0119390, -0.0100635, -0.0118364, -0.0100851, -0.0011004, 0.0009764
9: -0.0031415, -0.0029797, -0.0031396, -0.0029885, -0.0000842, 0.0000949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002390, upper bound: 0.0002452
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002390, upper bound: 0.0002507
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0129058, -0.0112439, -0.0127790, -0.0112439, -0.0009934, 0.0008549
1: -0.0065773, -0.0061087, -0.0065415, -0.0061087, -0.0002801, 0.0002410
2: -0.0099689, -0.0065118, -0.0097051, -0.0065118, -0.0020664, 0.0017784
3: 0.0003081, 0.0007656, 0.0003430, 0.0007656, -0.0002735, 0.0002353
4: 0.0109584, 0.0135420, 0.0109584, 0.0133448, -0.0013291, 0.0015443
5: 0.9985508, 0.9992685, 0.9985508, 0.9992138, -0.0003693, 0.0004291
6: 0.0065682, 0.0072198, 0.0065682, 0.0071700, -0.0003352, 0.0003894
7: 0.0011299, 0.0035614, 0.0011299, 0.0033758, -0.0012508, 0.0014534
8: -0.0119647, -0.0100723, -0.0118203, -0.0100723, -0.0011311, 0.0009735
9: -0.0031407, -0.0029775, -0.0031407, -0.0029899, -0.0000840, 0.0000976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002501
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002351, upper bound: 0.0002552
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0129058, -0.0112439, -0.0128004, -0.0112547, -0.0009761, 0.0008698
1: -0.0065773, -0.0061087, -0.0065476, -0.0061118, -0.0002752, 0.0002452
2: -0.0099689, -0.0065118, -0.0097497, -0.0065342, -0.0020305, 0.0018093
3: 0.0003081, 0.0007656, 0.0003371, 0.0007626, -0.0002687, 0.0002394
4: 0.0109584, 0.0135420, 0.0109751, 0.0133782, -0.0013522, 0.0015175
5: 0.9985508, 0.9992685, 0.9985554, 0.9992231, -0.0003757, 0.0004216
6: 0.0065682, 0.0072198, 0.0065724, 0.0071784, -0.0003410, 0.0003827
7: 0.0011299, 0.0035614, 0.0011456, 0.0034072, -0.0012726, 0.0014281
8: -0.0119647, -0.0100723, -0.0118447, -0.0100845, -0.0011115, 0.0009904
9: -0.0031407, -0.0029775, -0.0031397, -0.0029878, -0.0000854, 0.0000959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002501
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002351, upper bound: 0.0002552
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128729, -0.0112367, -0.0128742, -0.0112332, -0.0008145, 0.0008141
1: -0.0065680, -0.0061067, -0.0065684, -0.0061057, -0.0002296, 0.0002295
2: -0.0099005, -0.0064968, -0.0099032, -0.0064894, -0.0016944, 0.0016934
3: 0.0003171, 0.0007676, 0.0003168, 0.0007685, -0.0002242, 0.0002241
4: 0.0109471, 0.0134909, 0.0109416, 0.0134929, -0.0012656, 0.0012663
5: 0.9985477, 0.9992545, 0.9985461, 0.9992550, -0.0003516, 0.0003518
6: 0.0065654, 0.0072069, 0.0065640, 0.0072074, -0.0003192, 0.0003193
7: 0.0011194, 0.0035133, 0.0011142, 0.0035151, -0.0011910, 0.0011917
8: -0.0119272, -0.0100641, -0.0119287, -0.0100600, -0.0009275, 0.0009270
9: -0.0031415, -0.0029807, -0.0031418, -0.0029806, -0.0000800, 0.0000800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002223, upper bound: 0.0002138
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002452
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128833, -0.0112362, -0.0128986, -0.0112444, -0.0008243, 0.0008221
1: -0.0065709, -0.0061066, -0.0065753, -0.0061089, -0.0002324, 0.0002318
2: -0.0099219, -0.0064958, -0.0099539, -0.0065128, -0.0017148, 0.0017101
3: 0.0003143, 0.0007677, 0.0003101, 0.0007654, -0.0002269, 0.0002263
4: 0.0109464, 0.0135069, 0.0109591, 0.0135308, -0.0012781, 0.0012815
5: 0.9985474, 0.9992589, 0.9985510, 0.9992654, -0.0003551, 0.0003560
6: 0.0065652, 0.0072109, 0.0065684, 0.0072169, -0.0003223, 0.0003232
7: 0.0011186, 0.0035284, 0.0011306, 0.0035508, -0.0012028, 0.0012061
8: -0.0119390, -0.0100635, -0.0119565, -0.0100728, -0.0009387, 0.0009361
9: -0.0031415, -0.0029797, -0.0031407, -0.0029782, -0.0000808, 0.0000810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002392, upper bound: 0.0002452
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002392, upper bound: 0.0002506
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0129058, -0.0112439, -0.0128833, -0.0112362, -0.0008577, 0.0008248
1: -0.0065773, -0.0061087, -0.0065709, -0.0061066, -0.0002418, 0.0002325
2: -0.0099689, -0.0065118, -0.0099219, -0.0064958, -0.0017843, 0.0017157
3: 0.0003081, 0.0007656, 0.0003143, 0.0007677, -0.0002361, 0.0002270
4: 0.0109584, 0.0135420, 0.0109464, 0.0135069, -0.0012822, 0.0013334
5: 0.9985508, 0.9992685, 0.9985474, 0.9992589, -0.0003562, 0.0003705
6: 0.0065682, 0.0072198, 0.0065652, 0.0072109, -0.0003234, 0.0003363
7: 0.0011299, 0.0035614, 0.0011186, 0.0035284, -0.0012067, 0.0012549
8: -0.0119647, -0.0100723, -0.0119390, -0.0100635, -0.0009767, 0.0009392
9: -0.0031407, -0.0029775, -0.0031415, -0.0029797, -0.0000810, 0.0000843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002346, upper bound: 0.0002501
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002352, upper bound: 0.0002551
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0129058, -0.0112439, -0.0129058, -0.0112439, -0.0008340, 0.0008340
1: -0.0065773, -0.0061087, -0.0065773, -0.0061087, -0.0002351, 0.0002351
2: -0.0099689, -0.0065118, -0.0099689, -0.0065118, -0.0017348, 0.0017348
3: 0.0003081, 0.0007656, 0.0003081, 0.0007656, -0.0002296, 0.0002296
4: 0.0109584, 0.0135420, 0.0109584, 0.0135420, -0.0012965, 0.0012965
5: 0.9985508, 0.9992685, 0.9985508, 0.9992685, -0.0003602, 0.0003602
6: 0.0065682, 0.0072198, 0.0065682, 0.0072198, -0.0003270, 0.0003270
7: 0.0011299, 0.0035614, 0.0011299, 0.0035614, -0.0012201, 0.0012201
8: -0.0119647, -0.0100723, -0.0119647, -0.0100723, -0.0009496, 0.0009496
9: -0.0031407, -0.0029775, -0.0031407, -0.0029775, -0.0000819, 0.0000819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002346, upper bound: 0.0002501
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002352, upper bound: 0.0002552
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002198, upper bound: 0.0001971
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002305
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002394, upper bound: 0.0002329
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002394, upper bound: 0.0002360
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002374
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002360, upper bound: 0.0002400
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002353, upper bound: 0.0002375
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002360, upper bound: 0.0002399
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002340, upper bound: 0.0001970
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002452, upper bound: 0.0002297
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002546, upper bound: 0.0002323
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002546, upper bound: 0.0002352
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002372
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002507, upper bound: 0.0002397
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002501, upper bound: 0.0002371
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002507, upper bound: 0.0002396
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002197, upper bound: 0.0002136
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002450
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002390, upper bound: 0.0002452
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002390, upper bound: 0.0002507
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002501
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002351, upper bound: 0.0002552
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002345, upper bound: 0.0002501
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002351, upper bound: 0.0002552
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002223, upper bound: 0.0002138
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002452
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002392, upper bound: 0.0002452
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002392, upper bound: 0.0002506
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002346, upper bound: 0.0002501
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002352, upper bound: 0.0002551
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002346, upper bound: 0.0002501
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 5, lower bound: -0.0002352, upper bound: 0.0002552

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127614, -0.0112448, -0.0127720, -0.0112376, -0.0007221, 0.0007776
1: -0.0065366, -0.0061090, -0.0065396, -0.0061070, -0.0002036, 0.0002192
2: -0.0096684, -0.0065135, -0.0096905, -0.0064987, -0.0015021, 0.0016176
3: 0.0003478, 0.0007653, 0.0003449, 0.0007673, -0.0001988, 0.0002141
4: 0.0109597, 0.0133174, 0.0109486, 0.0133339, -0.0012089, 0.0011226
5: 0.9985511, 0.9992062, 0.9985481, 0.9992108, -0.0003359, 0.0003119
6: 0.0065685, 0.0071631, 0.0065657, 0.0071673, -0.0003049, 0.0002831
7: 0.0011311, 0.0033501, 0.0011207, 0.0033656, -0.0011377, 0.0010565
8: -0.0118002, -0.0100732, -0.0118123, -0.0100651, -0.0008222, 0.0008855
9: -0.0031407, -0.0029917, -0.0031414, -0.0029906, -0.0000764, 0.0000709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002306
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002305
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127449, -0.0112295, -0.0127932, -0.0112552, -0.0007474, 0.0008106
1: -0.0065319, -0.0061047, -0.0065455, -0.0061119, -0.0002107, 0.0002285
2: -0.0096342, -0.0064817, -0.0097346, -0.0065352, -0.0015548, 0.0016861
3: 0.0003524, 0.0007695, 0.0003391, 0.0007625, -0.0002058, 0.0002231
4: 0.0109359, 0.0132919, 0.0109759, 0.0133669, -0.0012601, 0.0011620
5: 0.9985445, 0.9991992, 0.9985557, 0.9992199, -0.0003501, 0.0003228
6: 0.0065625, 0.0071567, 0.0065726, 0.0071756, -0.0003178, 0.0002930
7: 0.0011088, 0.0033260, 0.0011464, 0.0033966, -0.0011859, 0.0010936
8: -0.0117815, -0.0100558, -0.0118364, -0.0100851, -0.0008511, 0.0009230
9: -0.0031422, -0.0029933, -0.0031396, -0.0029885, -0.0000796, 0.0000734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002329
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002329
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127677, -0.0112445, -0.0127932, -0.0112552, -0.0007553, 0.0007838
1: -0.0065384, -0.0061089, -0.0065455, -0.0061119, -0.0002129, 0.0002210
2: -0.0096816, -0.0065131, -0.0097346, -0.0065352, -0.0015711, 0.0016305
3: 0.0003461, 0.0007654, 0.0003391, 0.0007625, -0.0002079, 0.0002158
4: 0.0109593, 0.0133273, 0.0109759, 0.0133669, -0.0012185, 0.0011742
5: 0.9985510, 0.9992089, 0.9985557, 0.9992199, -0.0003385, 0.0003262
6: 0.0065684, 0.0071656, 0.0065726, 0.0071756, -0.0003073, 0.0002961
7: 0.0011308, 0.0033593, 0.0011464, 0.0033966, -0.0011467, 0.0011050
8: -0.0118074, -0.0100730, -0.0118364, -0.0100851, -0.0008600, 0.0008925
9: -0.0031407, -0.0029910, -0.0031396, -0.0029885, -0.0000770, 0.0000742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002360
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002360
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0127687, -0.0112445, -0.0007827, 0.0007791
1: -0.0065385, -0.0061070, -0.0065386, -0.0061089, -0.0002207, 0.0002197
2: -0.0096824, -0.0064989, -0.0096836, -0.0065129, -0.0016281, 0.0016207
3: 0.0003460, 0.0007673, 0.0003458, 0.0007654, -0.0002155, 0.0002145
4: 0.0109487, 0.0133279, 0.0109592, 0.0133288, -0.0012112, 0.0012167
5: 0.9985481, 0.9992091, 0.9985511, 0.9992094, -0.0003365, 0.0003380
6: 0.0065658, 0.0071658, 0.0065684, 0.0071660, -0.0003055, 0.0003068
7: 0.0011208, 0.0033599, 0.0011307, 0.0033608, -0.0011399, 0.0011451
8: -0.0118079, -0.0100652, -0.0118085, -0.0100729, -0.0008912, 0.0008872
9: -0.0031414, -0.0029910, -0.0031407, -0.0029910, -0.0000765, 0.0000769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001971, upper bound: 0.0002198
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002305, upper bound: 0.0002327
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0127790, -0.0112439, -0.0007886, 0.0007879
1: -0.0065440, -0.0061119, -0.0065415, -0.0061087, -0.0002223, 0.0002221
2: -0.0097235, -0.0065354, -0.0097051, -0.0065118, -0.0016405, 0.0016390
3: 0.0003405, 0.0007624, 0.0003430, 0.0007656, -0.0002171, 0.0002169
4: 0.0109760, 0.0133586, 0.0109584, 0.0133448, -0.0012249, 0.0012260
5: 0.9985557, 0.9992176, 0.9985508, 0.9992138, -0.0003403, 0.0003406
6: 0.0065727, 0.0071735, 0.0065682, 0.0071700, -0.0003089, 0.0003092
7: 0.0011465, 0.0033888, 0.0011299, 0.0033758, -0.0011528, 0.0011538
8: -0.0118304, -0.0100852, -0.0118203, -0.0100723, -0.0008980, 0.0008972
9: -0.0031396, -0.0029891, -0.0031407, -0.0029899, -0.0000774, 0.0000775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002329, upper bound: 0.0002393
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002329, upper bound: 0.0002399
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0127910, -0.0112552, -0.0007583, 0.0007881
1: -0.0065385, -0.0061070, -0.0065449, -0.0061119, -0.0002138, 0.0002222
2: -0.0096824, -0.0064989, -0.0097299, -0.0065353, -0.0015774, 0.0016394
3: 0.0003460, 0.0007673, 0.0003397, 0.0007625, -0.0002087, 0.0002169
4: 0.0109487, 0.0133279, 0.0109759, 0.0133634, -0.0012252, 0.0011788
5: 0.9985481, 0.9992091, 0.9985557, 0.9992189, -0.0003404, 0.0003275
6: 0.0065658, 0.0071658, 0.0065726, 0.0071747, -0.0003090, 0.0002973
7: 0.0011208, 0.0033599, 0.0011464, 0.0033933, -0.0011530, 0.0011094
8: -0.0118079, -0.0100652, -0.0118339, -0.0100851, -0.0008635, 0.0008974
9: -0.0031414, -0.0029910, -0.0031396, -0.0029888, -0.0000774, 0.0000745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001995, upper bound: 0.0002201
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002304, upper bound: 0.0002327
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0128004, -0.0112547, -0.0007649, 0.0007985
1: -0.0065440, -0.0061119, -0.0065476, -0.0061118, -0.0002157, 0.0002251
2: -0.0097235, -0.0065354, -0.0097497, -0.0065342, -0.0015912, 0.0016610
3: 0.0003405, 0.0007624, 0.0003371, 0.0007626, -0.0002106, 0.0002198
4: 0.0109760, 0.0133586, 0.0109751, 0.0133782, -0.0012413, 0.0011891
5: 0.9985557, 0.9992176, 0.9985554, 0.9992231, -0.0003449, 0.0003304
6: 0.0065727, 0.0071735, 0.0065724, 0.0071784, -0.0003130, 0.0002999
7: 0.0011465, 0.0033888, 0.0011456, 0.0034072, -0.0011682, 0.0011191
8: -0.0118304, -0.0100852, -0.0118447, -0.0100845, -0.0008710, 0.0009092
9: -0.0031396, -0.0029891, -0.0031397, -0.0029878, -0.0000784, 0.0000751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002394
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002400
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0126882, -0.0112058, -0.0128589, -0.0112339, -0.0007518, 0.0009582
1: -0.0065159, -0.0060980, -0.0065640, -0.0061059, -0.0002120, 0.0002701
2: -0.0095163, -0.0064324, -0.0098712, -0.0064910, -0.0015639, 0.0019932
3: 0.0003680, 0.0007761, 0.0003210, 0.0007683, -0.0002070, 0.0002638
4: 0.0108990, 0.0132037, 0.0109429, 0.0134690, -0.0014896, 0.0011688
5: 0.9985343, 0.9991746, 0.9985465, 0.9992483, -0.0004138, 0.0003247
6: 0.0065532, 0.0071345, 0.0065643, 0.0072013, -0.0003756, 0.0002947
7: 0.0010741, 0.0032431, 0.0011153, 0.0034927, -0.0014018, 0.0010999
8: -0.0117169, -0.0100288, -0.0119112, -0.0100609, -0.0008561, 0.0010911
9: -0.0031445, -0.0029989, -0.0031417, -0.0029821, -0.0000941, 0.0000739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002246, upper bound: 0.0001970
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002246, upper bound: 0.0001970
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127614, -0.0112448, -0.0128742, -0.0112332, -0.0007954, 0.0009525
1: -0.0065366, -0.0061090, -0.0065684, -0.0061057, -0.0002242, 0.0002686
2: -0.0096684, -0.0065135, -0.0099032, -0.0064894, -0.0016545, 0.0019814
3: 0.0003478, 0.0007653, 0.0003168, 0.0007685, -0.0002189, 0.0002622
4: 0.0109597, 0.0133174, 0.0109416, 0.0134929, -0.0014808, 0.0012365
5: 0.9985511, 0.9992062, 0.9985461, 0.9992550, -0.0004114, 0.0003435
6: 0.0065685, 0.0071631, 0.0065640, 0.0072074, -0.0003734, 0.0003118
7: 0.0011311, 0.0033501, 0.0011142, 0.0035151, -0.0013936, 0.0011637
8: -0.0118002, -0.0100732, -0.0119287, -0.0100600, -0.0009057, 0.0010846
9: -0.0031407, -0.0029917, -0.0031418, -0.0029806, -0.0000936, 0.0000781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002297
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002297
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127449, -0.0112295, -0.0128986, -0.0112444, -0.0008140, 0.0009865
1: -0.0065319, -0.0061047, -0.0065753, -0.0061089, -0.0002295, 0.0002781
2: -0.0096342, -0.0064817, -0.0099539, -0.0065128, -0.0016933, 0.0020522
3: 0.0003524, 0.0007695, 0.0003101, 0.0007654, -0.0002241, 0.0002716
4: 0.0109359, 0.0132919, 0.0109591, 0.0135308, -0.0015337, 0.0012655
5: 0.9985445, 0.9991992, 0.9985510, 0.9992654, -0.0004261, 0.0003516
6: 0.0065625, 0.0071567, 0.0065684, 0.0072169, -0.0003868, 0.0003191
7: 0.0011088, 0.0033260, 0.0011306, 0.0035508, -0.0014434, 0.0011909
8: -0.0117815, -0.0100558, -0.0119565, -0.0100728, -0.0009269, 0.0011234
9: -0.0031422, -0.0029933, -0.0031407, -0.0029782, -0.0000969, 0.0000800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002323
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002323
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127677, -0.0112445, -0.0128986, -0.0112444, -0.0008243, 0.0009587
1: -0.0065384, -0.0061089, -0.0065753, -0.0061089, -0.0002324, 0.0002703
2: -0.0096816, -0.0065131, -0.0099539, -0.0065128, -0.0017147, 0.0019944
3: 0.0003461, 0.0007654, 0.0003101, 0.0007654, -0.0002269, 0.0002639
4: 0.0109593, 0.0133273, 0.0109591, 0.0135308, -0.0014905, 0.0012814
5: 0.9985510, 0.9992089, 0.9985510, 0.9992654, -0.0004141, 0.0003560
6: 0.0065684, 0.0071656, 0.0065684, 0.0072169, -0.0003759, 0.0003232
7: 0.0011308, 0.0033593, 0.0011306, 0.0035508, -0.0014027, 0.0012060
8: -0.0118074, -0.0100730, -0.0119565, -0.0100728, -0.0009386, 0.0010917
9: -0.0031407, -0.0029910, -0.0031407, -0.0029782, -0.0000942, 0.0000810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002351
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002351
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0128729, -0.0112367, -0.0008534, 0.0009570
1: -0.0065385, -0.0061070, -0.0065680, -0.0061067, -0.0002406, 0.0002698
2: -0.0096824, -0.0064989, -0.0099005, -0.0064968, -0.0017752, 0.0019907
3: 0.0003460, 0.0007673, 0.0003171, 0.0007676, -0.0002349, 0.0002634
4: 0.0109487, 0.0133279, 0.0109471, 0.0134909, -0.0014877, 0.0013267
5: 0.9985481, 0.9992091, 0.9985477, 0.9992545, -0.0004133, 0.0003686
6: 0.0065658, 0.0071658, 0.0065654, 0.0072069, -0.0003752, 0.0003346
7: 0.0011208, 0.0033599, 0.0011194, 0.0035133, -0.0014001, 0.0012486
8: -0.0118079, -0.0100652, -0.0119272, -0.0100641, -0.0009718, 0.0010897
9: -0.0031414, -0.0029910, -0.0031415, -0.0029807, -0.0000940, 0.0000838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002135, upper bound: 0.0002197
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002451, upper bound: 0.0002324
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0128833, -0.0112362, -0.0008617, 0.0009662
1: -0.0065440, -0.0061119, -0.0065709, -0.0061066, -0.0002430, 0.0002724
2: -0.0097235, -0.0065354, -0.0099219, -0.0064958, -0.0017926, 0.0020100
3: 0.0003405, 0.0007624, 0.0003143, 0.0007677, -0.0002372, 0.0002660
4: 0.0109760, 0.0133586, 0.0109464, 0.0135069, -0.0015021, 0.0013397
5: 0.9985557, 0.9992176, 0.9985474, 0.9992589, -0.0004173, 0.0003722
6: 0.0065727, 0.0071735, 0.0065652, 0.0072109, -0.0003788, 0.0003378
7: 0.0011465, 0.0033888, 0.0011186, 0.0035284, -0.0014137, 0.0012608
8: -0.0118304, -0.0100852, -0.0119390, -0.0100635, -0.0009813, 0.0011003
9: -0.0031396, -0.0029891, -0.0031415, -0.0029797, -0.0000949, 0.0000847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002451, upper bound: 0.0002391
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002451, upper bound: 0.0002397
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0128954, -0.0112444, -0.0008291, 0.0009649
1: -0.0065385, -0.0061070, -0.0065744, -0.0061089, -0.0002338, 0.0002720
2: -0.0096824, -0.0064989, -0.0099472, -0.0065129, -0.0017247, 0.0020072
3: 0.0003460, 0.0007673, 0.0003109, 0.0007654, -0.0002282, 0.0002656
4: 0.0109487, 0.0133279, 0.0109592, 0.0135258, -0.0015001, 0.0012889
5: 0.9985481, 0.9992091, 0.9985510, 0.9992641, -0.0004168, 0.0003581
6: 0.0065658, 0.0071658, 0.0065684, 0.0072157, -0.0003783, 0.0003251
7: 0.0011208, 0.0033599, 0.0011306, 0.0035462, -0.0014117, 0.0012130
8: -0.0118079, -0.0100652, -0.0119528, -0.0100729, -0.0009441, 0.0010988
9: -0.0031414, -0.0029910, -0.0031407, -0.0029785, -0.0000948, 0.0000815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002168, upper bound: 0.0002200
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002324
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0129058, -0.0112439, -0.0008375, 0.0009756
1: -0.0065440, -0.0061119, -0.0065773, -0.0061087, -0.0002361, 0.0002750
2: -0.0097235, -0.0065354, -0.0099689, -0.0065118, -0.0017422, 0.0020294
3: 0.0003405, 0.0007624, 0.0003081, 0.0007656, -0.0002306, 0.0002686
4: 0.0109760, 0.0133586, 0.0109584, 0.0135420, -0.0015166, 0.0013020
5: 0.9985557, 0.9992176, 0.9985508, 0.9992685, -0.0004214, 0.0003617
6: 0.0065727, 0.0071735, 0.0065682, 0.0072198, -0.0003825, 0.0003283
7: 0.0011465, 0.0033888, 0.0011299, 0.0035614, -0.0014273, 0.0012253
8: -0.0118304, -0.0100852, -0.0119647, -0.0100723, -0.0009537, 0.0011109
9: -0.0031396, -0.0029891, -0.0031407, -0.0029775, -0.0000958, 0.0000823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002452, upper bound: 0.0002391
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002452, upper bound: 0.0002397
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128650, -0.0112370, -0.0127720, -0.0112376, -0.0009067, 0.0008483
1: -0.0065658, -0.0061068, -0.0065396, -0.0061070, -0.0002556, 0.0002392
2: -0.0098840, -0.0064975, -0.0096905, -0.0064987, -0.0018862, 0.0017647
3: 0.0003193, 0.0007675, 0.0003449, 0.0007673, -0.0002496, 0.0002335
4: 0.0109477, 0.0134785, 0.0109486, 0.0133339, -0.0013188, 0.0014096
5: 0.9985479, 0.9992510, 0.9985481, 0.9992108, -0.0003664, 0.0003916
6: 0.0065655, 0.0072038, 0.0065657, 0.0071673, -0.0003326, 0.0003555
7: 0.0011198, 0.0035017, 0.0011207, 0.0033656, -0.0012411, 0.0013266
8: -0.0119182, -0.0100644, -0.0118123, -0.0100651, -0.0010325, 0.0009660
9: -0.0031414, -0.0029815, -0.0031414, -0.0029906, -0.0000833, 0.0000891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002452
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002451
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128478, -0.0112248, -0.0127932, -0.0112552, -0.0009245, 0.0008834
1: -0.0065609, -0.0061033, -0.0065455, -0.0061119, -0.0002607, 0.0002491
2: -0.0098481, -0.0064719, -0.0097346, -0.0065352, -0.0019232, 0.0018376
3: 0.0003241, 0.0007708, 0.0003391, 0.0007625, -0.0002545, 0.0002432
4: 0.0109286, 0.0134517, 0.0109759, 0.0133669, -0.0013733, 0.0014373
5: 0.9985425, 0.9992435, 0.9985557, 0.9992199, -0.0003815, 0.0003993
6: 0.0065607, 0.0071970, 0.0065726, 0.0071756, -0.0003463, 0.0003625
7: 0.0011019, 0.0034764, 0.0011464, 0.0033966, -0.0012924, 0.0013527
8: -0.0118986, -0.0100505, -0.0118364, -0.0100851, -0.0010528, 0.0010059
9: -0.0031426, -0.0029832, -0.0031396, -0.0029885, -0.0000868, 0.0000908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002452
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002452
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128724, -0.0112368, -0.0127932, -0.0112552, -0.0009325, 0.0008570
1: -0.0065679, -0.0061067, -0.0065455, -0.0061119, -0.0002629, 0.0002416
2: -0.0098994, -0.0064969, -0.0097346, -0.0065352, -0.0019399, 0.0017828
3: 0.0003173, 0.0007675, 0.0003391, 0.0007625, -0.0002567, 0.0002359
4: 0.0109473, 0.0134900, 0.0109759, 0.0133669, -0.0013323, 0.0014497
5: 0.9985477, 0.9992542, 0.9985557, 0.9992199, -0.0003702, 0.0004028
6: 0.0065654, 0.0072067, 0.0065726, 0.0071756, -0.0003360, 0.0003656
7: 0.0011195, 0.0035125, 0.0011464, 0.0033966, -0.0012539, 0.0013644
8: -0.0119266, -0.0100641, -0.0118364, -0.0100851, -0.0010619, 0.0009759
9: -0.0031415, -0.0029808, -0.0031396, -0.0029885, -0.0000842, 0.0000916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002507
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002507
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0127687, -0.0112445, -0.0009515, 0.0008479
1: -0.0065672, -0.0061057, -0.0065386, -0.0061089, -0.0002683, 0.0002390
2: -0.0098946, -0.0064896, -0.0096836, -0.0065129, -0.0019793, 0.0017638
3: 0.0003179, 0.0007685, 0.0003458, 0.0007654, -0.0002619, 0.0002334
4: 0.0109418, 0.0134865, 0.0109592, 0.0133288, -0.0013181, 0.0014792
5: 0.9985462, 0.9992533, 0.9985511, 0.9992094, -0.0003662, 0.0004110
6: 0.0065640, 0.0072058, 0.0065684, 0.0071660, -0.0003324, 0.0003730
7: 0.0011143, 0.0035092, 0.0011307, 0.0033608, -0.0012405, 0.0013921
8: -0.0119241, -0.0100601, -0.0118085, -0.0100729, -0.0010835, 0.0009655
9: -0.0031418, -0.0029810, -0.0031407, -0.0029910, -0.0000833, 0.0000935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001970, upper bound: 0.0002340
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0127790, -0.0112439, -0.0009578, 0.0008544
1: -0.0065742, -0.0061089, -0.0065415, -0.0061087, -0.0002700, 0.0002409
2: -0.0099462, -0.0065130, -0.0097051, -0.0065118, -0.0019924, 0.0017774
3: 0.0003111, 0.0007654, 0.0003430, 0.0007656, -0.0002637, 0.0002352
4: 0.0109593, 0.0135251, 0.0109584, 0.0133448, -0.0013283, 0.0014890
5: 0.9985511, 0.9992639, 0.9985508, 0.9992138, -0.0003690, 0.0004137
6: 0.0065684, 0.0072155, 0.0065682, 0.0071700, -0.0003350, 0.0003755
7: 0.0011308, 0.0035454, 0.0011299, 0.0033758, -0.0012501, 0.0014013
8: -0.0119523, -0.0100729, -0.0118203, -0.0100723, -0.0010906, 0.0009730
9: -0.0031407, -0.0029786, -0.0031407, -0.0029899, -0.0000839, 0.0000941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002547
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002550
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0127910, -0.0112552, -0.0009346, 0.0008607
1: -0.0065672, -0.0061057, -0.0065449, -0.0061119, -0.0002635, 0.0002427
2: -0.0098946, -0.0064896, -0.0097299, -0.0065353, -0.0019442, 0.0017905
3: 0.0003179, 0.0007685, 0.0003397, 0.0007625, -0.0002573, 0.0002369
4: 0.0109418, 0.0134865, 0.0109759, 0.0133634, -0.0013381, 0.0014530
5: 0.9985462, 0.9992533, 0.9985557, 0.9992189, -0.0003718, 0.0004037
6: 0.0065640, 0.0072058, 0.0065726, 0.0071747, -0.0003375, 0.0003664
7: 0.0011143, 0.0035092, 0.0011464, 0.0033933, -0.0012593, 0.0013674
8: -0.0119241, -0.0100601, -0.0118339, -0.0100851, -0.0010643, 0.0009801
9: -0.0031418, -0.0029810, -0.0031396, -0.0029888, -0.0000846, 0.0000918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001995, upper bound: 0.0002343
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0128004, -0.0112547, -0.0009413, 0.0008693
1: -0.0065742, -0.0061089, -0.0065476, -0.0061118, -0.0002654, 0.0002451
2: -0.0099462, -0.0065130, -0.0097497, -0.0065342, -0.0019580, 0.0018083
3: 0.0003111, 0.0007654, 0.0003371, 0.0007626, -0.0002591, 0.0002393
4: 0.0109593, 0.0135251, 0.0109751, 0.0133782, -0.0013514, 0.0014633
5: 0.9985511, 0.9992639, 0.9985554, 0.9992231, -0.0003755, 0.0004066
6: 0.0065684, 0.0072155, 0.0065724, 0.0071784, -0.0003408, 0.0003690
7: 0.0011308, 0.0035454, 0.0011456, 0.0034072, -0.0012719, 0.0013771
8: -0.0119523, -0.0100729, -0.0118447, -0.0100845, -0.0010718, 0.0009899
9: -0.0031407, -0.0029786, -0.0031397, -0.0029878, -0.0000854, 0.0000925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002547
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002552
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128650, -0.0112370, -0.0128742, -0.0112332, -0.0007546, 0.0008137
1: -0.0065658, -0.0061068, -0.0065684, -0.0061057, -0.0002127, 0.0002294
2: -0.0098840, -0.0064975, -0.0099032, -0.0064894, -0.0015697, 0.0016926
3: 0.0003193, 0.0007675, 0.0003168, 0.0007685, -0.0002077, 0.0002240
4: 0.0109477, 0.0134785, 0.0109416, 0.0134929, -0.0012649, 0.0011731
5: 0.9985479, 0.9992510, 0.9985461, 0.9992550, -0.0003514, 0.0003259
6: 0.0065655, 0.0072038, 0.0065640, 0.0072074, -0.0003190, 0.0002958
7: 0.0011198, 0.0035017, 0.0011142, 0.0035151, -0.0011905, 0.0011040
8: -0.0119182, -0.0100644, -0.0119287, -0.0100600, -0.0008593, 0.0009265
9: -0.0031414, -0.0029815, -0.0031418, -0.0029806, -0.0000799, 0.0000741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002296, upper bound: 0.0002451
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002296, upper bound: 0.0002452
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128478, -0.0112248, -0.0128986, -0.0112444, -0.0007836, 0.0008466
1: -0.0065609, -0.0061033, -0.0065753, -0.0061089, -0.0002209, 0.0002387
2: -0.0098481, -0.0064719, -0.0099539, -0.0065128, -0.0016301, 0.0017612
3: 0.0003241, 0.0007708, 0.0003101, 0.0007654, -0.0002157, 0.0002331
4: 0.0109286, 0.0134517, 0.0109591, 0.0135308, -0.0013162, 0.0012182
5: 0.9985425, 0.9992435, 0.9985510, 0.9992654, -0.0003657, 0.0003385
6: 0.0065607, 0.0071970, 0.0065684, 0.0072169, -0.0003319, 0.0003072
7: 0.0011019, 0.0034764, 0.0011306, 0.0035508, -0.0012387, 0.0011465
8: -0.0118986, -0.0100505, -0.0119565, -0.0100728, -0.0008923, 0.0009641
9: -0.0031426, -0.0029832, -0.0031407, -0.0029782, -0.0000832, 0.0000770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002452
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002451
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128724, -0.0112368, -0.0128986, -0.0112444, -0.0007931, 0.0008216
1: -0.0065679, -0.0061067, -0.0065753, -0.0061089, -0.0002236, 0.0002316
2: -0.0098994, -0.0064969, -0.0099539, -0.0065128, -0.0016498, 0.0017091
3: 0.0003173, 0.0007675, 0.0003101, 0.0007654, -0.0002183, 0.0002262
4: 0.0109473, 0.0134900, 0.0109591, 0.0135308, -0.0012773, 0.0012330
5: 0.9985477, 0.9992542, 0.9985510, 0.9992654, -0.0003549, 0.0003426
6: 0.0065654, 0.0072067, 0.0065684, 0.0072169, -0.0003221, 0.0003109
7: 0.0011195, 0.0035125, 0.0011306, 0.0035508, -0.0012021, 0.0011604
8: -0.0119266, -0.0100641, -0.0119565, -0.0100728, -0.0009031, 0.0009356
9: -0.0031415, -0.0029808, -0.0031407, -0.0029782, -0.0000807, 0.0000779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002506
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002506
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0128729, -0.0112367, -0.0008190, 0.0008144
1: -0.0065672, -0.0061057, -0.0065680, -0.0061067, -0.0002309, 0.0002296
2: -0.0098946, -0.0064896, -0.0099005, -0.0064968, -0.0017037, 0.0016942
3: 0.0003179, 0.0007685, 0.0003171, 0.0007676, -0.0002255, 0.0002242
4: 0.0109418, 0.0134865, 0.0109471, 0.0134909, -0.0012661, 0.0012732
5: 0.9985462, 0.9992533, 0.9985477, 0.9992545, -0.0003518, 0.0003537
6: 0.0065640, 0.0072058, 0.0065654, 0.0072069, -0.0003193, 0.0003211
7: 0.0011143, 0.0035092, 0.0011194, 0.0035133, -0.0011916, 0.0011982
8: -0.0119241, -0.0100601, -0.0119272, -0.0100641, -0.0009326, 0.0009274
9: -0.0031418, -0.0029810, -0.0031415, -0.0029807, -0.0000800, 0.0000805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002012, upper bound: 0.0002346
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002454
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0128833, -0.0112362, -0.0008267, 0.0008243
1: -0.0065742, -0.0061089, -0.0065709, -0.0061066, -0.0002331, 0.0002324
2: -0.0099462, -0.0065130, -0.0099219, -0.0064958, -0.0017197, 0.0017146
3: 0.0003111, 0.0007654, 0.0003143, 0.0007677, -0.0002276, 0.0002269
4: 0.0109593, 0.0135251, 0.0109464, 0.0135069, -0.0012814, 0.0012852
5: 0.9985511, 0.9992639, 0.9985474, 0.9992589, -0.0003560, 0.0003571
6: 0.0065684, 0.0072155, 0.0065652, 0.0072109, -0.0003232, 0.0003241
7: 0.0011308, 0.0035454, 0.0011186, 0.0035284, -0.0012060, 0.0012095
8: -0.0119523, -0.0100729, -0.0119390, -0.0100635, -0.0009413, 0.0009386
9: -0.0031407, -0.0029786, -0.0031415, -0.0029797, -0.0000810, 0.0000812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002546
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002551
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0128954, -0.0112444, -0.0007930, 0.0008218
1: -0.0065672, -0.0061057, -0.0065744, -0.0061089, -0.0002236, 0.0002317
2: -0.0098946, -0.0064896, -0.0099472, -0.0065129, -0.0016496, 0.0017095
3: 0.0003179, 0.0007685, 0.0003109, 0.0007654, -0.0002183, 0.0002262
4: 0.0109418, 0.0134865, 0.0109592, 0.0135258, -0.0012775, 0.0012328
5: 0.9985462, 0.9992533, 0.9985510, 0.9992641, -0.0003549, 0.0003425
6: 0.0065640, 0.0072058, 0.0065684, 0.0072157, -0.0003222, 0.0003109
7: 0.0011143, 0.0035092, 0.0011306, 0.0035462, -0.0012023, 0.0011602
8: -0.0119241, -0.0100601, -0.0119528, -0.0100729, -0.0009030, 0.0009358
9: -0.0031418, -0.0029810, -0.0031407, -0.0029785, -0.0000807, 0.0000779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002034, upper bound: 0.0002348
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002296, upper bound: 0.0002453
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0129058, -0.0112439, -0.0008015, 0.0008335
1: -0.0065742, -0.0061089, -0.0065773, -0.0061087, -0.0002260, 0.0002350
2: -0.0099462, -0.0065130, -0.0099689, -0.0065118, -0.0016674, 0.0017338
3: 0.0003111, 0.0007654, 0.0003081, 0.0007656, -0.0002207, 0.0002294
4: 0.0109593, 0.0135251, 0.0109584, 0.0135420, -0.0012957, 0.0012461
5: 0.9985511, 0.9992639, 0.9985508, 0.9992685, -0.0003600, 0.0003462
6: 0.0065684, 0.0072155, 0.0065682, 0.0072198, -0.0003268, 0.0003142
7: 0.0011308, 0.0035454, 0.0011299, 0.0035614, -0.0012194, 0.0011727
8: -0.0119523, -0.0100729, -0.0119647, -0.0100723, -0.0009127, 0.0009491
9: -0.0031407, -0.0029786, -0.0031407, -0.0029775, -0.0000819, 0.0000787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002325, upper bound: 0.0002547
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002325, upper bound: 0.0002552
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.03 seconds
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002306
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002305
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002329
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002329
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002360
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002360
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0001971, upper bound: 0.0002198
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002305, upper bound: 0.0002327
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002329, upper bound: 0.0002393
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002329, upper bound: 0.0002399
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0001995, upper bound: 0.0002201
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002304, upper bound: 0.0002327
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002394
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002330, upper bound: 0.0002400
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002246, upper bound: 0.0001970
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002246, upper bound: 0.0001970
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002297
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002421, upper bound: 0.0002297
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002323
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002323
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002351
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002471, upper bound: 0.0002351
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002135, upper bound: 0.0002197
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002451, upper bound: 0.0002324
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002451, upper bound: 0.0002391
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002451, upper bound: 0.0002397
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002168, upper bound: 0.0002200
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002450, upper bound: 0.0002324
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002452, upper bound: 0.0002391
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002452, upper bound: 0.0002397
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002452
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002451
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002452
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002452
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002507
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002341, upper bound: 0.0002507
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0001970, upper bound: 0.0002340
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002547
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002550
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0001995, upper bound: 0.0002343
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002547
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002323, upper bound: 0.0002552
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002296, upper bound: 0.0002451
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002296, upper bound: 0.0002452
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002452
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002451
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002506
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002344, upper bound: 0.0002506
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002012, upper bound: 0.0002346
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002454
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002546
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002551
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002034, upper bound: 0.0002348
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002296, upper bound: 0.0002453
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002325, upper bound: 0.0002547
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 5, lower bound: -0.0002325, upper bound: 0.0002552

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127614, -0.0112448, -0.0127449, -0.0112295, -0.0007212, 0.0007475
1: -0.0065366, -0.0061090, -0.0065319, -0.0061047, -0.0002033, 0.0002108
2: -0.0096684, -0.0065135, -0.0096342, -0.0064817, -0.0015003, 0.0015550
3: 0.0003478, 0.0007653, 0.0003524, 0.0007695, -0.0001985, 0.0002058
4: 0.0109597, 0.0133174, 0.0109359, 0.0132919, -0.0011621, 0.0011212
5: 0.9985511, 0.9992062, 0.9985445, 0.9991992, -0.0003229, 0.0003115
6: 0.0065685, 0.0071631, 0.0065625, 0.0071567, -0.0002931, 0.0002828
7: 0.0011311, 0.0033501, 0.0011088, 0.0033260, -0.0010937, 0.0010552
8: -0.0118002, -0.0100732, -0.0117815, -0.0100558, -0.0008213, 0.0008512
9: -0.0031407, -0.0029917, -0.0031422, -0.0029933, -0.0000734, 0.0000709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002292
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002305
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127614, -0.0112448, -0.0127681, -0.0112377, -0.0007220, 0.0007823
1: -0.0065366, -0.0061090, -0.0065385, -0.0061070, -0.0002036, 0.0002206
2: -0.0096684, -0.0065135, -0.0096824, -0.0064989, -0.0015019, 0.0016274
3: 0.0003478, 0.0007653, 0.0003460, 0.0007673, -0.0001988, 0.0002154
4: 0.0109597, 0.0133174, 0.0109487, 0.0133279, -0.0012162, 0.0011224
5: 0.9985511, 0.9992062, 0.9985481, 0.9992091, -0.0003379, 0.0003118
6: 0.0065685, 0.0071631, 0.0065658, 0.0071658, -0.0003067, 0.0002831
7: 0.0011311, 0.0033501, 0.0011208, 0.0033599, -0.0011446, 0.0010563
8: -0.0118002, -0.0100732, -0.0118079, -0.0100652, -0.0008222, 0.0008908
9: -0.0031407, -0.0029917, -0.0031414, -0.0029910, -0.0000769, 0.0000709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002292
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002304
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127449, -0.0112295, -0.0127677, -0.0112445, -0.0007478, 0.0007811
1: -0.0065319, -0.0061047, -0.0065384, -0.0061089, -0.0002108, 0.0002202
2: -0.0096342, -0.0064817, -0.0096816, -0.0065131, -0.0015556, 0.0016248
3: 0.0003524, 0.0007695, 0.0003461, 0.0007654, -0.0002059, 0.0002150
4: 0.0109359, 0.0132919, 0.0109593, 0.0133273, -0.0012143, 0.0011626
5: 0.9985445, 0.9991992, 0.9985510, 0.9992089, -0.0003374, 0.0003230
6: 0.0065625, 0.0071567, 0.0065684, 0.0071656, -0.0003062, 0.0002932
7: 0.0011088, 0.0033260, 0.0011308, 0.0033593, -0.0011428, 0.0010941
8: -0.0117815, -0.0100558, -0.0118074, -0.0100730, -0.0008515, 0.0008894
9: -0.0031422, -0.0029933, -0.0031407, -0.0029910, -0.0000767, 0.0000735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001814
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002282
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127449, -0.0112295, -0.0127878, -0.0112553, -0.0007474, 0.0008131
1: -0.0065319, -0.0061047, -0.0065440, -0.0061119, -0.0002107, 0.0002293
2: -0.0096342, -0.0064817, -0.0097235, -0.0065354, -0.0015547, 0.0016915
3: 0.0003524, 0.0007695, 0.0003405, 0.0007624, -0.0002057, 0.0002238
4: 0.0109359, 0.0132919, 0.0109760, 0.0133586, -0.0012641, 0.0011619
5: 0.9985445, 0.9991992, 0.9985557, 0.9992176, -0.0003512, 0.0003228
6: 0.0065625, 0.0071567, 0.0065727, 0.0071735, -0.0003188, 0.0002930
7: 0.0011088, 0.0033260, 0.0011465, 0.0033888, -0.0011897, 0.0010935
8: -0.0117815, -0.0100558, -0.0118304, -0.0100852, -0.0008510, 0.0009259
9: -0.0031422, -0.0029933, -0.0031396, -0.0029891, -0.0000799, 0.0000734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001814
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002282
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127677, -0.0112445, -0.0127677, -0.0112445, -0.0007536, 0.0007536
1: -0.0065384, -0.0061089, -0.0065384, -0.0061089, -0.0002125, 0.0002125
2: -0.0096816, -0.0065131, -0.0096816, -0.0065131, -0.0015677, 0.0015677
3: 0.0003461, 0.0007654, 0.0003461, 0.0007654, -0.0002075, 0.0002075
4: 0.0109593, 0.0133273, 0.0109593, 0.0133273, -0.0011716, 0.0011716
5: 0.9985510, 0.9992089, 0.9985510, 0.9992089, -0.0003255, 0.0003255
6: 0.0065684, 0.0071656, 0.0065684, 0.0071656, -0.0002955, 0.0002955
7: 0.0011308, 0.0033593, 0.0011308, 0.0033593, -0.0011026, 0.0011026
8: -0.0118074, -0.0100730, -0.0118074, -0.0100730, -0.0008582, 0.0008582
9: -0.0031407, -0.0029910, -0.0031407, -0.0029910, -0.0000740, 0.0000740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002156
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002301, upper bound: 0.0002307
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127677, -0.0112445, -0.0127878, -0.0112553, -0.0007552, 0.0007881
1: -0.0065384, -0.0061089, -0.0065440, -0.0061119, -0.0002129, 0.0002222
2: -0.0096816, -0.0065131, -0.0097235, -0.0065354, -0.0015709, 0.0016393
3: 0.0003461, 0.0007654, 0.0003405, 0.0007624, -0.0002079, 0.0002169
4: 0.0109593, 0.0133273, 0.0109760, 0.0133586, -0.0012251, 0.0011740
5: 0.9985510, 0.9992089, 0.9985557, 0.9992176, -0.0003404, 0.0003262
6: 0.0065684, 0.0071656, 0.0065727, 0.0071735, -0.0003090, 0.0002961
7: 0.0011308, 0.0033593, 0.0011465, 0.0033888, -0.0011530, 0.0011049
8: -0.0118074, -0.0100730, -0.0118304, -0.0100852, -0.0008599, 0.0008974
9: -0.0031407, -0.0029910, -0.0031396, -0.0029891, -0.0000774, 0.0000742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002157
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002301, upper bound: 0.0002306
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0127614, -0.0112448, -0.0007823, 0.0007220
1: -0.0065385, -0.0061070, -0.0065366, -0.0061090, -0.0002206, 0.0002036
2: -0.0096824, -0.0064989, -0.0096684, -0.0065135, -0.0016274, 0.0015019
3: 0.0003460, 0.0007673, 0.0003478, 0.0007653, -0.0002154, 0.0001988
4: 0.0109487, 0.0133279, 0.0109597, 0.0133174, -0.0011224, 0.0012162
5: 0.9985481, 0.9992091, 0.9985511, 0.9992062, -0.0003118, 0.0003379
6: 0.0065658, 0.0071658, 0.0065685, 0.0071631, -0.0002831, 0.0003067
7: 0.0011208, 0.0033599, 0.0011311, 0.0033501, -0.0010563, 0.0011446
8: -0.0118079, -0.0100652, -0.0118002, -0.0100732, -0.0008908, 0.0008222
9: -0.0031414, -0.0029910, -0.0031407, -0.0029917, -0.0000709, 0.0000769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0001840
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0002327
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0127449, -0.0112295, -0.0008131, 0.0007474
1: -0.0065440, -0.0061119, -0.0065319, -0.0061047, -0.0002293, 0.0002107
2: -0.0097235, -0.0065354, -0.0096342, -0.0064817, -0.0016915, 0.0015547
3: 0.0003405, 0.0007624, 0.0003524, 0.0007695, -0.0002238, 0.0002057
4: 0.0109760, 0.0133586, 0.0109359, 0.0132919, -0.0011619, 0.0012641
5: 0.9985557, 0.9992176, 0.9985445, 0.9991992, -0.0003228, 0.0003512
6: 0.0065727, 0.0071735, 0.0065625, 0.0071567, -0.0002930, 0.0003188
7: 0.0011465, 0.0033888, 0.0011088, 0.0033260, -0.0010935, 0.0011897
8: -0.0118304, -0.0100852, -0.0117815, -0.0100558, -0.0009259, 0.0008510
9: -0.0031396, -0.0029891, -0.0031422, -0.0029933, -0.0000734, 0.0000799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002040
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002282, upper bound: 0.0002345
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0127677, -0.0112445, -0.0007881, 0.0007552
1: -0.0065440, -0.0061119, -0.0065384, -0.0061089, -0.0002222, 0.0002129
2: -0.0097235, -0.0065354, -0.0096816, -0.0065131, -0.0016393, 0.0015709
3: 0.0003405, 0.0007624, 0.0003461, 0.0007654, -0.0002169, 0.0002079
4: 0.0109760, 0.0133586, 0.0109593, 0.0133273, -0.0011740, 0.0012251
5: 0.9985557, 0.9992176, 0.9985510, 0.9992089, -0.0003262, 0.0003404
6: 0.0065727, 0.0071735, 0.0065684, 0.0071656, -0.0002961, 0.0003090
7: 0.0011465, 0.0033888, 0.0011308, 0.0033593, -0.0011049, 0.0011530
8: -0.0118304, -0.0100852, -0.0118074, -0.0100730, -0.0008974, 0.0008599
9: -0.0031396, -0.0029891, -0.0031407, -0.0029910, -0.0000742, 0.0000774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002209
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002282, upper bound: 0.0002346
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0127838, -0.0112555, -0.0007579, 0.0007307
1: -0.0065385, -0.0061070, -0.0065429, -0.0061120, -0.0002137, 0.0002060
2: -0.0096824, -0.0064989, -0.0097151, -0.0065359, -0.0015767, 0.0015200
3: 0.0003460, 0.0007673, 0.0003417, 0.0007624, -0.0002086, 0.0002012
4: 0.0109487, 0.0133279, 0.0109764, 0.0133523, -0.0011360, 0.0011783
5: 0.9985481, 0.9992091, 0.9985558, 0.9992158, -0.0003156, 0.0003274
6: 0.0065658, 0.0071658, 0.0065727, 0.0071719, -0.0002865, 0.0002972
7: 0.0011208, 0.0033599, 0.0011468, 0.0033829, -0.0010691, 0.0011089
8: -0.0118079, -0.0100652, -0.0118258, -0.0100855, -0.0008631, 0.0008321
9: -0.0031414, -0.0029910, -0.0031396, -0.0029895, -0.0000718, 0.0000745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0001850
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0002327
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0127681, -0.0112377, -0.0007916, 0.0007582
1: -0.0065440, -0.0061119, -0.0065385, -0.0061070, -0.0002232, 0.0002138
2: -0.0097235, -0.0065354, -0.0096824, -0.0064989, -0.0016466, 0.0015772
3: 0.0003405, 0.0007624, 0.0003460, 0.0007673, -0.0002179, 0.0002087
4: 0.0109760, 0.0133586, 0.0109487, 0.0133279, -0.0011787, 0.0012306
5: 0.9985557, 0.9992176, 0.9985481, 0.9992091, -0.0003275, 0.0003419
6: 0.0065727, 0.0071735, 0.0065658, 0.0071658, -0.0002973, 0.0003103
7: 0.0011465, 0.0033888, 0.0011208, 0.0033599, -0.0011093, 0.0011581
8: -0.0118304, -0.0100852, -0.0118079, -0.0100652, -0.0009014, 0.0008634
9: -0.0031396, -0.0029891, -0.0031414, -0.0029910, -0.0000745, 0.0000778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002046
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002281, upper bound: 0.0002345
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0127878, -0.0112553, -0.0007644, 0.0007644
1: -0.0065440, -0.0061119, -0.0065440, -0.0061119, -0.0002155, 0.0002155
2: -0.0097235, -0.0065354, -0.0097235, -0.0065354, -0.0015900, 0.0015900
3: 0.0003405, 0.0007624, 0.0003405, 0.0007624, -0.0002104, 0.0002104
4: 0.0109760, 0.0133586, 0.0109760, 0.0133586, -0.0011883, 0.0011883
5: 0.9985557, 0.9992176, 0.9985557, 0.9992176, -0.0003301, 0.0003301
6: 0.0065727, 0.0071735, 0.0065727, 0.0071735, -0.0002997, 0.0002997
7: 0.0011465, 0.0033888, 0.0011465, 0.0033888, -0.0011183, 0.0011183
8: -0.0118304, -0.0100852, -0.0118304, -0.0100852, -0.0008704, 0.0008704
9: -0.0031396, -0.0029891, -0.0031396, -0.0029891, -0.0000751, 0.0000751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002211
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002281, upper bound: 0.0002346
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127614, -0.0112448, -0.0128478, -0.0112248, -0.0007991, 0.0009246
1: -0.0065366, -0.0061090, -0.0065609, -0.0061033, -0.0002253, 0.0002607
2: -0.0096684, -0.0065135, -0.0098481, -0.0064719, -0.0016623, 0.0019234
3: 0.0003478, 0.0007653, 0.0003241, 0.0007708, -0.0002200, 0.0002545
4: 0.0109597, 0.0133174, 0.0109286, 0.0134517, -0.0014374, 0.0012423
5: 0.9985511, 0.9992062, 0.9985425, 0.9992435, -0.0003994, 0.0003451
6: 0.0065685, 0.0071631, 0.0065607, 0.0071970, -0.0003625, 0.0003133
7: 0.0011311, 0.0033501, 0.0011019, 0.0034764, -0.0013528, 0.0011691
8: -0.0118002, -0.0100732, -0.0118986, -0.0100505, -0.0009099, 0.0010529
9: -0.0031407, -0.0029917, -0.0031426, -0.0029832, -0.0000908, 0.0000785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002286
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002297
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127614, -0.0112448, -0.0128701, -0.0112332, -0.0007953, 0.0009511
1: -0.0065366, -0.0061090, -0.0065672, -0.0061057, -0.0002242, 0.0002682
2: -0.0096684, -0.0065135, -0.0098946, -0.0064896, -0.0016543, 0.0019786
3: 0.0003478, 0.0007653, 0.0003179, 0.0007685, -0.0002189, 0.0002618
4: 0.0109597, 0.0133174, 0.0109418, 0.0134865, -0.0014787, 0.0012363
5: 0.9985511, 0.9992062, 0.9985462, 0.9992533, -0.0004108, 0.0003435
6: 0.0065685, 0.0071631, 0.0065640, 0.0072058, -0.0003729, 0.0003118
7: 0.0011311, 0.0033501, 0.0011143, 0.0035092, -0.0013916, 0.0011635
8: -0.0118002, -0.0100732, -0.0119241, -0.0100601, -0.0009056, 0.0010831
9: -0.0031407, -0.0029917, -0.0031418, -0.0029810, -0.0000934, 0.0000781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002285
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002297
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0127449, -0.0112295, -0.0128724, -0.0112368, -0.0008185, 0.0009597
1: -0.0065319, -0.0061047, -0.0065679, -0.0061067, -0.0002308, 0.0002706
2: -0.0096342, -0.0064817, -0.0098994, -0.0064969, -0.0017027, 0.0019963
3: 0.0003524, 0.0007695, 0.0003173, 0.0007675, -0.0002253, 0.0002642
4: 0.0109359, 0.0132919, 0.0109473, 0.0134900, -0.0014919, 0.0012725
5: 0.9985445, 0.9991992, 0.9985477, 0.9992542, -0.0004145, 0.0003535
6: 0.0065625, 0.0071567, 0.0065654, 0.0072067, -0.0003762, 0.0003209
7: 0.0011088, 0.0033260, 0.0011195, 0.0035125, -0.0014041, 0.0011976
8: -0.0117815, -0.0100558, -0.0119266, -0.0100641, -0.0009321, 0.0010928
9: -0.0031422, -0.0029933, -0.0031415, -0.0029808, -0.0000943, 0.0000804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002343, upper bound: 0.0001814
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002470, upper bound: 0.0002275
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127449, -0.0112295, -0.0128949, -0.0112445, -0.0008139, 0.0009855
1: -0.0065319, -0.0061047, -0.0065742, -0.0061089, -0.0002295, 0.0002778
2: -0.0096342, -0.0064817, -0.0099462, -0.0065130, -0.0016931, 0.0020499
3: 0.0003524, 0.0007695, 0.0003111, 0.0007654, -0.0002241, 0.0002713
4: 0.0109359, 0.0132919, 0.0109593, 0.0135251, -0.0015320, 0.0012653
5: 0.9985445, 0.9991992, 0.9985511, 0.9992639, -0.0004256, 0.0003515
6: 0.0065625, 0.0071567, 0.0065684, 0.0072155, -0.0003863, 0.0003191
7: 0.0011088, 0.0033260, 0.0011308, 0.0035454, -0.0014418, 0.0011908
8: -0.0117815, -0.0100558, -0.0119523, -0.0100729, -0.0009268, 0.0011221
9: -0.0031422, -0.0029933, -0.0031407, -0.0029786, -0.0000968, 0.0000800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002343, upper bound: 0.0001814
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002470, upper bound: 0.0002275
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127677, -0.0112445, -0.0128724, -0.0112368, -0.0008268, 0.0009309
1: -0.0065384, -0.0061089, -0.0065679, -0.0061067, -0.0002331, 0.0002625
2: -0.0096816, -0.0065131, -0.0098994, -0.0064969, -0.0017200, 0.0019365
3: 0.0003461, 0.0007654, 0.0003173, 0.0007675, -0.0002276, 0.0002563
4: 0.0109593, 0.0133273, 0.0109473, 0.0134900, -0.0014472, 0.0012854
5: 0.9985510, 0.9992089, 0.9985477, 0.9992542, -0.0004021, 0.0003571
6: 0.0065684, 0.0071656, 0.0065654, 0.0072067, -0.0003650, 0.0003242
7: 0.0011308, 0.0033593, 0.0011195, 0.0035125, -0.0013620, 0.0012097
8: -0.0118074, -0.0100730, -0.0119266, -0.0100641, -0.0009415, 0.0010600
9: -0.0031407, -0.0029910, -0.0031415, -0.0029808, -0.0000915, 0.0000812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002405, upper bound: 0.0002156
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002297
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127677, -0.0112445, -0.0128949, -0.0112445, -0.0008242, 0.0009572
1: -0.0065384, -0.0061089, -0.0065742, -0.0061089, -0.0002324, 0.0002699
2: -0.0096816, -0.0065131, -0.0099462, -0.0065130, -0.0017145, 0.0019912
3: 0.0003461, 0.0007654, 0.0003111, 0.0007654, -0.0002269, 0.0002635
4: 0.0109593, 0.0133273, 0.0109593, 0.0135251, -0.0014881, 0.0012813
5: 0.9985510, 0.9992089, 0.9985511, 0.9992639, -0.0004134, 0.0003560
6: 0.0065684, 0.0071656, 0.0065684, 0.0072155, -0.0003753, 0.0003231
7: 0.0011308, 0.0033593, 0.0011308, 0.0035454, -0.0014005, 0.0012058
8: -0.0118074, -0.0100730, -0.0119523, -0.0100729, -0.0009385, 0.0010900
9: -0.0031407, -0.0029910, -0.0031407, -0.0029786, -0.0000940, 0.0000810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002405, upper bound: 0.0002156
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002298
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0128650, -0.0112370, -0.0008530, 0.0009066
1: -0.0065385, -0.0061070, -0.0065658, -0.0061068, -0.0002405, 0.0002556
2: -0.0096824, -0.0064989, -0.0098840, -0.0064975, -0.0017745, 0.0018860
3: 0.0003460, 0.0007673, 0.0003193, 0.0007675, -0.0002348, 0.0002496
4: 0.0109487, 0.0133279, 0.0109477, 0.0134785, -0.0014095, 0.0013261
5: 0.9985481, 0.9992091, 0.9985479, 0.9992510, -0.0003916, 0.0003684
6: 0.0065658, 0.0071658, 0.0065655, 0.0072038, -0.0003555, 0.0003344
7: 0.0011208, 0.0033599, 0.0011198, 0.0035017, -0.0013265, 0.0012480
8: -0.0118079, -0.0100652, -0.0119182, -0.0100644, -0.0009713, 0.0010324
9: -0.0031414, -0.0029910, -0.0031414, -0.0029815, -0.0000891, 0.0000838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002312, upper bound: 0.0001839
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002312, upper bound: 0.0002324
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0128478, -0.0112248, -0.0008860, 0.0009245
1: -0.0065440, -0.0061119, -0.0065609, -0.0061033, -0.0002498, 0.0002606
2: -0.0097235, -0.0065354, -0.0098481, -0.0064719, -0.0018430, 0.0019231
3: 0.0003405, 0.0007624, 0.0003241, 0.0007708, -0.0002439, 0.0002545
4: 0.0109760, 0.0133586, 0.0109286, 0.0134517, -0.0014372, 0.0013773
5: 0.9985557, 0.9992176, 0.9985425, 0.9992435, -0.0003993, 0.0003827
6: 0.0065727, 0.0071735, 0.0065607, 0.0071970, -0.0003624, 0.0003473
7: 0.0011465, 0.0033888, 0.0011019, 0.0034764, -0.0013525, 0.0012962
8: -0.0118304, -0.0100852, -0.0118986, -0.0100505, -0.0010088, 0.0010527
9: -0.0031396, -0.0029891, -0.0031426, -0.0029832, -0.0000908, 0.0000870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002224, upper bound: 0.0002040
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002342
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0128724, -0.0112368, -0.0008613, 0.0009325
1: -0.0065440, -0.0061119, -0.0065679, -0.0061067, -0.0002428, 0.0002629
2: -0.0097235, -0.0065354, -0.0098994, -0.0064969, -0.0017916, 0.0019397
3: 0.0003405, 0.0007624, 0.0003173, 0.0007675, -0.0002371, 0.0002567
4: 0.0109760, 0.0133586, 0.0109473, 0.0134900, -0.0014496, 0.0013390
5: 0.9985557, 0.9992176, 0.9985477, 0.9992542, -0.0004027, 0.0003720
6: 0.0065727, 0.0071735, 0.0065654, 0.0072067, -0.0003656, 0.0003377
7: 0.0011465, 0.0033888, 0.0011195, 0.0035125, -0.0013642, 0.0012601
8: -0.0118304, -0.0100852, -0.0119266, -0.0100641, -0.0009807, 0.0010618
9: -0.0031396, -0.0029891, -0.0031415, -0.0029808, -0.0000916, 0.0000846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002224, upper bound: 0.0002209
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002341
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0127681, -0.0112377, -0.0128878, -0.0112447, -0.0008287, 0.0009159
1: -0.0065385, -0.0061070, -0.0065722, -0.0061090, -0.0002336, 0.0002582
2: -0.0096824, -0.0064989, -0.0099315, -0.0065135, -0.0017239, 0.0019052
3: 0.0003460, 0.0007673, 0.0003130, 0.0007653, -0.0002281, 0.0002521
4: 0.0109487, 0.0133279, 0.0109597, 0.0135141, -0.0014238, 0.0012884
5: 0.9985481, 0.9992091, 0.9985511, 0.9992608, -0.0003956, 0.0003579
6: 0.0065658, 0.0071658, 0.0065685, 0.0072127, -0.0003591, 0.0003249
7: 0.0011208, 0.0033599, 0.0011311, 0.0035351, -0.0013400, 0.0012125
8: -0.0118079, -0.0100652, -0.0119442, -0.0100732, -0.0009437, 0.0010429
9: -0.0031414, -0.0029910, -0.0031407, -0.0029792, -0.0000900, 0.0000814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0001850
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002323
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0128701, -0.0112332, -0.0008642, 0.0009345
1: -0.0065440, -0.0061119, -0.0065672, -0.0061057, -0.0002437, 0.0002635
2: -0.0097235, -0.0065354, -0.0098946, -0.0064896, -0.0017978, 0.0019440
3: 0.0003405, 0.0007624, 0.0003179, 0.0007685, -0.0002379, 0.0002573
4: 0.0109760, 0.0133586, 0.0109418, 0.0134865, -0.0014529, 0.0013435
5: 0.9985557, 0.9992176, 0.9985462, 0.9992533, -0.0004036, 0.0003733
6: 0.0065727, 0.0071735, 0.0065640, 0.0072058, -0.0003664, 0.0003388
7: 0.0011465, 0.0033888, 0.0011143, 0.0035092, -0.0013673, 0.0012644
8: -0.0118304, -0.0100852, -0.0119241, -0.0100601, -0.0009841, 0.0010642
9: -0.0031396, -0.0029891, -0.0031418, -0.0029810, -0.0000918, 0.0000849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002243, upper bound: 0.0002046
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002342
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127878, -0.0112553, -0.0128949, -0.0112445, -0.0008370, 0.0009407
1: -0.0065440, -0.0061119, -0.0065742, -0.0061089, -0.0002360, 0.0002652
2: -0.0097235, -0.0065354, -0.0099462, -0.0065130, -0.0017412, 0.0019569
3: 0.0003405, 0.0007624, 0.0003111, 0.0007654, -0.0002304, 0.0002590
4: 0.0109760, 0.0133586, 0.0109593, 0.0135251, -0.0014625, 0.0013012
5: 0.9985557, 0.9992176, 0.9985511, 0.9992639, -0.0004063, 0.0003615
6: 0.0065727, 0.0071735, 0.0065684, 0.0072155, -0.0003688, 0.0003282
7: 0.0011465, 0.0033888, 0.0011308, 0.0035454, -0.0013763, 0.0012246
8: -0.0118304, -0.0100852, -0.0119523, -0.0100729, -0.0009531, 0.0010712
9: -0.0031396, -0.0029891, -0.0031407, -0.0029786, -0.0000924, 0.0000822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002243, upper bound: 0.0002211
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002343
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128650, -0.0112370, -0.0127449, -0.0112295, -0.0009059, 0.0008182
1: -0.0065658, -0.0061068, -0.0065319, -0.0061047, -0.0002554, 0.0002307
2: -0.0098840, -0.0064975, -0.0096342, -0.0064817, -0.0018844, 0.0017021
3: 0.0003193, 0.0007675, 0.0003524, 0.0007695, -0.0002494, 0.0002252
4: 0.0109477, 0.0134785, 0.0109359, 0.0132919, -0.0012720, 0.0014083
5: 0.9985479, 0.9992510, 0.9985445, 0.9991992, -0.0003534, 0.0003913
6: 0.0065655, 0.0072038, 0.0065625, 0.0071567, -0.0003208, 0.0003552
7: 0.0011198, 0.0035017, 0.0011088, 0.0033260, -0.0011971, 0.0013254
8: -0.0119182, -0.0100644, -0.0117815, -0.0100558, -0.0010315, 0.0009317
9: -0.0031414, -0.0029815, -0.0031422, -0.0029933, -0.0000804, 0.0000890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002429
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002451
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128650, -0.0112370, -0.0127681, -0.0112377, -0.0009066, 0.0008530
1: -0.0065658, -0.0061068, -0.0065385, -0.0061070, -0.0002556, 0.0002405
2: -0.0098840, -0.0064975, -0.0096824, -0.0064989, -0.0018860, 0.0017745
3: 0.0003193, 0.0007675, 0.0003460, 0.0007673, -0.0002496, 0.0002348
4: 0.0109477, 0.0134785, 0.0109487, 0.0133279, -0.0013261, 0.0014095
5: 0.9985479, 0.9992510, 0.9985481, 0.9992091, -0.0003684, 0.0003916
6: 0.0065655, 0.0072038, 0.0065658, 0.0071658, -0.0003344, 0.0003555
7: 0.0011198, 0.0035017, 0.0011208, 0.0033599, -0.0012480, 0.0013265
8: -0.0119182, -0.0100644, -0.0118079, -0.0100652, -0.0010324, 0.0009713
9: -0.0031414, -0.0029815, -0.0031414, -0.0029910, -0.0000838, 0.0000891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002429
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002451
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128478, -0.0112248, -0.0127677, -0.0112445, -0.0009249, 0.0008539
1: -0.0065609, -0.0061033, -0.0065384, -0.0061089, -0.0002608, 0.0002407
2: -0.0098481, -0.0064719, -0.0096816, -0.0065131, -0.0019240, 0.0017763
3: 0.0003241, 0.0007708, 0.0003461, 0.0007654, -0.0002546, 0.0002351
4: 0.0109286, 0.0134517, 0.0109593, 0.0133273, -0.0013275, 0.0014379
5: 0.9985425, 0.9992435, 0.9985510, 0.9992089, -0.0003688, 0.0003995
6: 0.0065607, 0.0071970, 0.0065684, 0.0071656, -0.0003348, 0.0003626
7: 0.0011019, 0.0034764, 0.0011308, 0.0033593, -0.0012493, 0.0013532
8: -0.0118986, -0.0100505, -0.0118074, -0.0100730, -0.0010532, 0.0009723
9: -0.0031426, -0.0029832, -0.0031407, -0.0029910, -0.0000839, 0.0000909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001951
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002403
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128478, -0.0112248, -0.0127878, -0.0112553, -0.0009245, 0.0008860
1: -0.0065609, -0.0061033, -0.0065440, -0.0061119, -0.0002606, 0.0002498
2: -0.0098481, -0.0064719, -0.0097235, -0.0065354, -0.0019231, 0.0018430
3: 0.0003241, 0.0007708, 0.0003405, 0.0007624, -0.0002545, 0.0002439
4: 0.0109286, 0.0134517, 0.0109760, 0.0133586, -0.0013773, 0.0014372
5: 0.9985425, 0.9992435, 0.9985557, 0.9992176, -0.0003827, 0.0003993
6: 0.0065607, 0.0071970, 0.0065727, 0.0071735, -0.0003473, 0.0003624
7: 0.0011019, 0.0034764, 0.0011465, 0.0033888, -0.0012962, 0.0013525
8: -0.0118986, -0.0100505, -0.0118304, -0.0100852, -0.0010527, 0.0010088
9: -0.0031426, -0.0029832, -0.0031396, -0.0029891, -0.0000870, 0.0000908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001951
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002404
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128724, -0.0112368, -0.0127677, -0.0112445, -0.0009309, 0.0008268
1: -0.0065679, -0.0061067, -0.0065384, -0.0061089, -0.0002625, 0.0002331
2: -0.0098994, -0.0064969, -0.0096816, -0.0065131, -0.0019365, 0.0017200
3: 0.0003173, 0.0007675, 0.0003461, 0.0007654, -0.0002563, 0.0002276
4: 0.0109473, 0.0134900, 0.0109593, 0.0133273, -0.0012854, 0.0014472
5: 0.9985477, 0.9992542, 0.9985510, 0.9992089, -0.0003571, 0.0004021
6: 0.0065654, 0.0072067, 0.0065684, 0.0071656, -0.0003242, 0.0003650
7: 0.0011195, 0.0035125, 0.0011308, 0.0033593, -0.0012097, 0.0013620
8: -0.0119266, -0.0100641, -0.0118074, -0.0100730, -0.0010600, 0.0009415
9: -0.0031415, -0.0029808, -0.0031407, -0.0029910, -0.0000812, 0.0000915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002333
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128724, -0.0112368, -0.0127878, -0.0112553, -0.0009325, 0.0008613
1: -0.0065679, -0.0061067, -0.0065440, -0.0061119, -0.0002629, 0.0002428
2: -0.0098994, -0.0064969, -0.0097235, -0.0065354, -0.0019397, 0.0017916
3: 0.0003173, 0.0007675, 0.0003405, 0.0007624, -0.0002567, 0.0002371
4: 0.0109473, 0.0134900, 0.0109760, 0.0133586, -0.0013390, 0.0014496
5: 0.9985477, 0.9992542, 0.9985557, 0.9992176, -0.0003720, 0.0004027
6: 0.0065654, 0.0072067, 0.0065727, 0.0071735, -0.0003377, 0.0003656
7: 0.0011195, 0.0035125, 0.0011465, 0.0033888, -0.0012601, 0.0013642
8: -0.0119266, -0.0100641, -0.0118304, -0.0100852, -0.0010618, 0.0009807
9: -0.0031415, -0.0029808, -0.0031396, -0.0029891, -0.0000846, 0.0000916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002333
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002451
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128545, -0.0112340, -0.0126882, -0.0112058, -0.0009571, 0.0007517
1: -0.0065628, -0.0061060, -0.0065159, -0.0060980, -0.0002698, 0.0002119
2: -0.0098621, -0.0064912, -0.0095163, -0.0064324, -0.0019910, 0.0015637
3: 0.0003222, 0.0007683, 0.0003680, 0.0007761, -0.0002635, 0.0002069
4: 0.0109430, 0.0134622, 0.0108990, 0.0132037, -0.0011686, 0.0014879
5: 0.9985466, 0.9992465, 0.9985343, 0.9991746, -0.0003247, 0.0004134
6: 0.0065643, 0.0071996, 0.0065532, 0.0071345, -0.0002947, 0.0003752
7: 0.0011154, 0.0034863, 0.0010741, 0.0032431, -0.0010998, 0.0014003
8: -0.0119063, -0.0100610, -0.0117169, -0.0100288, -0.0010899, 0.0008560
9: -0.0031417, -0.0029825, -0.0031445, -0.0029989, -0.0000738, 0.0000940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001922, upper bound: 0.0001947
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001922, upper bound: 0.0002340
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0127614, -0.0112448, -0.0009511, 0.0007953
1: -0.0065672, -0.0061057, -0.0065366, -0.0061090, -0.0002682, 0.0002242
2: -0.0098946, -0.0064896, -0.0096684, -0.0065135, -0.0019786, 0.0016543
3: 0.0003179, 0.0007685, 0.0003478, 0.0007653, -0.0002618, 0.0002189
4: 0.0109418, 0.0134865, 0.0109597, 0.0133174, -0.0012363, 0.0014787
5: 0.9985462, 0.9992533, 0.9985511, 0.9992062, -0.0003435, 0.0004108
6: 0.0065640, 0.0072058, 0.0065685, 0.0071631, -0.0003118, 0.0003729
7: 0.0011143, 0.0035092, 0.0011311, 0.0033501, -0.0011635, 0.0013916
8: -0.0119241, -0.0100601, -0.0118002, -0.0100732, -0.0010831, 0.0009056
9: -0.0031418, -0.0029810, -0.0031407, -0.0029917, -0.0000781, 0.0000934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0001972
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0002453
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0127449, -0.0112295, -0.0009855, 0.0008139
1: -0.0065742, -0.0061089, -0.0065319, -0.0061047, -0.0002778, 0.0002295
2: -0.0099462, -0.0065130, -0.0096342, -0.0064817, -0.0020499, 0.0016931
3: 0.0003111, 0.0007654, 0.0003524, 0.0007695, -0.0002713, 0.0002241
4: 0.0109593, 0.0135251, 0.0109359, 0.0132919, -0.0012653, 0.0015320
5: 0.9985511, 0.9992639, 0.9985445, 0.9991992, -0.0003515, 0.0004256
6: 0.0065684, 0.0072155, 0.0065625, 0.0071567, -0.0003191, 0.0003863
7: 0.0011308, 0.0035454, 0.0011088, 0.0033260, -0.0011908, 0.0014418
8: -0.0119523, -0.0100729, -0.0117815, -0.0100558, -0.0011221, 0.0009268
9: -0.0031407, -0.0029786, -0.0031422, -0.0029933, -0.0000800, 0.0000968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002202
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002497
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0127677, -0.0112445, -0.0009572, 0.0008242
1: -0.0065742, -0.0061089, -0.0065384, -0.0061089, -0.0002699, 0.0002324
2: -0.0099462, -0.0065130, -0.0096816, -0.0065131, -0.0019912, 0.0017145
3: 0.0003111, 0.0007654, 0.0003461, 0.0007654, -0.0002635, 0.0002269
4: 0.0109593, 0.0135251, 0.0109593, 0.0133273, -0.0012813, 0.0014881
5: 0.9985511, 0.9992639, 0.9985510, 0.9992089, -0.0003560, 0.0004134
6: 0.0065684, 0.0072155, 0.0065684, 0.0071656, -0.0003231, 0.0003753
7: 0.0011308, 0.0035454, 0.0011308, 0.0033593, -0.0012058, 0.0014005
8: -0.0119523, -0.0100729, -0.0118074, -0.0100730, -0.0010900, 0.0009385
9: -0.0031407, -0.0029786, -0.0031407, -0.0029910, -0.0000810, 0.0000940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002387
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002494
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128545, -0.0112340, -0.0127102, -0.0112166, -0.0009395, 0.0007632
1: -0.0065628, -0.0061060, -0.0065221, -0.0061010, -0.0002649, 0.0002152
2: -0.0098621, -0.0064912, -0.0095619, -0.0064550, -0.0019543, 0.0015877
3: 0.0003222, 0.0007683, 0.0003619, 0.0007731, -0.0002586, 0.0002101
4: 0.0109430, 0.0134622, 0.0109159, 0.0132378, -0.0011866, 0.0014605
5: 0.9985466, 0.9992465, 0.9985390, 0.9991841, -0.0003297, 0.0004058
6: 0.0065643, 0.0071996, 0.0065575, 0.0071431, -0.0002992, 0.0003683
7: 0.0011154, 0.0034863, 0.0010900, 0.0032751, -0.0011167, 0.0013745
8: -0.0119063, -0.0100610, -0.0117419, -0.0100412, -0.0010698, 0.0008691
9: -0.0031417, -0.0029825, -0.0031434, -0.0029967, -0.0000750, 0.0000923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001953, upper bound: 0.0001952
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001953, upper bound: 0.0002343
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0127838, -0.0112555, -0.0009343, 0.0008088
1: -0.0065672, -0.0061057, -0.0065429, -0.0061120, -0.0002634, 0.0002280
2: -0.0098946, -0.0064896, -0.0097151, -0.0065359, -0.0019435, 0.0016824
3: 0.0003179, 0.0007685, 0.0003417, 0.0007624, -0.0002572, 0.0002226
4: 0.0109418, 0.0134865, 0.0109764, 0.0133523, -0.0012573, 0.0014525
5: 0.9985462, 0.9992533, 0.9985558, 0.9992158, -0.0003493, 0.0004035
6: 0.0065640, 0.0072058, 0.0065727, 0.0071719, -0.0003171, 0.0003663
7: 0.0011143, 0.0035092, 0.0011468, 0.0033829, -0.0011833, 0.0013669
8: -0.0119241, -0.0100601, -0.0118258, -0.0100855, -0.0010639, 0.0009209
9: -0.0031418, -0.0029810, -0.0031396, -0.0029895, -0.0000795, 0.0000918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0001977
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0002453
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0127681, -0.0112377, -0.0009690, 0.0008290
1: -0.0065742, -0.0061089, -0.0065385, -0.0061070, -0.0002732, 0.0002337
2: -0.0099462, -0.0065130, -0.0096824, -0.0064989, -0.0020157, 0.0017246
3: 0.0003111, 0.0007654, 0.0003460, 0.0007673, -0.0002667, 0.0002282
4: 0.0109593, 0.0135251, 0.0109487, 0.0133279, -0.0012888, 0.0015064
5: 0.9985511, 0.9992639, 0.9985481, 0.9992091, -0.0003581, 0.0004185
6: 0.0065684, 0.0072155, 0.0065658, 0.0071658, -0.0003250, 0.0003799
7: 0.0011308, 0.0035454, 0.0011208, 0.0033599, -0.0012129, 0.0014177
8: -0.0119523, -0.0100729, -0.0118079, -0.0100652, -0.0011034, 0.0009440
9: -0.0031407, -0.0029786, -0.0031414, -0.0029910, -0.0000814, 0.0000952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002206
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002498
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0127878, -0.0112553, -0.0009407, 0.0008370
1: -0.0065742, -0.0061089, -0.0065440, -0.0061119, -0.0002652, 0.0002360
2: -0.0099462, -0.0065130, -0.0097235, -0.0065354, -0.0019569, 0.0017412
3: 0.0003111, 0.0007654, 0.0003405, 0.0007624, -0.0002590, 0.0002304
4: 0.0109593, 0.0135251, 0.0109760, 0.0133586, -0.0013012, 0.0014625
5: 0.9985511, 0.9992639, 0.9985557, 0.9992176, -0.0003615, 0.0004063
6: 0.0065684, 0.0072155, 0.0065727, 0.0071735, -0.0003282, 0.0003688
7: 0.0011308, 0.0035454, 0.0011465, 0.0033888, -0.0012246, 0.0013763
8: -0.0119523, -0.0100729, -0.0118304, -0.0100852, -0.0010712, 0.0009531
9: -0.0031407, -0.0029786, -0.0031396, -0.0029891, -0.0000822, 0.0000924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002389
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002498
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128650, -0.0112370, -0.0128478, -0.0112248, -0.0007551, 0.0007839
1: -0.0065658, -0.0061068, -0.0065609, -0.0061033, -0.0002129, 0.0002210
2: -0.0098840, -0.0064975, -0.0098481, -0.0064719, -0.0015708, 0.0016307
3: 0.0003193, 0.0007675, 0.0003241, 0.0007708, -0.0002079, 0.0002158
4: 0.0109477, 0.0134785, 0.0109286, 0.0134517, -0.0012187, 0.0011739
5: 0.9985479, 0.9992510, 0.9985425, 0.9992435, -0.0003386, 0.0003261
6: 0.0065655, 0.0072038, 0.0065607, 0.0071970, -0.0003073, 0.0002960
7: 0.0011198, 0.0035017, 0.0011019, 0.0034764, -0.0011469, 0.0011048
8: -0.0119182, -0.0100644, -0.0118986, -0.0100505, -0.0008598, 0.0008927
9: -0.0031414, -0.0029815, -0.0031426, -0.0029832, -0.0000770, 0.0000742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002429
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002451
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128650, -0.0112370, -0.0128701, -0.0112332, -0.0007545, 0.0008186
1: -0.0065658, -0.0061068, -0.0065672, -0.0061057, -0.0002127, 0.0002308
2: -0.0098840, -0.0064975, -0.0098946, -0.0064896, -0.0015695, 0.0017028
3: 0.0003193, 0.0007675, 0.0003179, 0.0007685, -0.0002077, 0.0002253
4: 0.0109477, 0.0134785, 0.0109418, 0.0134865, -0.0012726, 0.0011730
5: 0.9985479, 0.9992510, 0.9985462, 0.9992533, -0.0003536, 0.0003259
6: 0.0065655, 0.0072038, 0.0065640, 0.0072058, -0.0003209, 0.0002958
7: 0.0011198, 0.0035017, 0.0011143, 0.0035092, -0.0011977, 0.0011039
8: -0.0119182, -0.0100644, -0.0119241, -0.0100601, -0.0008592, 0.0009321
9: -0.0031414, -0.0029815, -0.0031418, -0.0029810, -0.0000804, 0.0000741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002429
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002452
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128478, -0.0112248, -0.0128724, -0.0112368, -0.0007843, 0.0008174
1: -0.0065609, -0.0061033, -0.0065679, -0.0061067, -0.0002211, 0.0002304
2: -0.0098481, -0.0064719, -0.0098994, -0.0064969, -0.0016314, 0.0017003
3: 0.0003241, 0.0007708, 0.0003173, 0.0007675, -0.0002159, 0.0002250
4: 0.0109286, 0.0134517, 0.0109473, 0.0134900, -0.0012707, 0.0012192
5: 0.9985425, 0.9992435, 0.9985477, 0.9992542, -0.0003530, 0.0003387
6: 0.0065607, 0.0071970, 0.0065654, 0.0072067, -0.0003204, 0.0003075
7: 0.0011019, 0.0034764, 0.0011195, 0.0035125, -0.0011959, 0.0011474
8: -0.0118986, -0.0100505, -0.0119266, -0.0100641, -0.0008930, 0.0009307
9: -0.0031426, -0.0029832, -0.0031415, -0.0029808, -0.0000803, 0.0000770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002212, upper bound: 0.0001959
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002404
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128478, -0.0112248, -0.0128949, -0.0112445, -0.0007835, 0.0008492
1: -0.0065609, -0.0061033, -0.0065742, -0.0061089, -0.0002209, 0.0002394
2: -0.0098481, -0.0064719, -0.0099462, -0.0065130, -0.0016299, 0.0017666
3: 0.0003241, 0.0007708, 0.0003111, 0.0007654, -0.0002157, 0.0002338
4: 0.0109286, 0.0134517, 0.0109593, 0.0135251, -0.0013202, 0.0012181
5: 0.9985425, 0.9992435, 0.9985511, 0.9992639, -0.0003668, 0.0003384
6: 0.0065607, 0.0071970, 0.0065684, 0.0072155, -0.0003329, 0.0003072
7: 0.0011019, 0.0034764, 0.0011308, 0.0035454, -0.0012425, 0.0011464
8: -0.0118986, -0.0100505, -0.0119523, -0.0100729, -0.0008922, 0.0009670
9: -0.0031426, -0.0029832, -0.0031407, -0.0029786, -0.0000834, 0.0000770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002212, upper bound: 0.0001959
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002402
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128724, -0.0112368, -0.0128724, -0.0112368, -0.0007922, 0.0007922
1: -0.0065679, -0.0061067, -0.0065679, -0.0061067, -0.0002233, 0.0002233
2: -0.0098994, -0.0064969, -0.0098994, -0.0064969, -0.0016479, 0.0016479
3: 0.0003173, 0.0007675, 0.0003173, 0.0007675, -0.0002181, 0.0002181
4: 0.0109473, 0.0134900, 0.0109473, 0.0134900, -0.0012315, 0.0012315
5: 0.9985477, 0.9992542, 0.9985477, 0.9992542, -0.0003422, 0.0003422
6: 0.0065654, 0.0072067, 0.0065654, 0.0072067, -0.0003106, 0.0003106
7: 0.0011195, 0.0035125, 0.0011195, 0.0035125, -0.0011590, 0.0011590
8: -0.0119266, -0.0100641, -0.0119266, -0.0100641, -0.0009021, 0.0009021
9: -0.0031415, -0.0029808, -0.0031415, -0.0029808, -0.0000778, 0.0000778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002284, upper bound: 0.0002336
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128724, -0.0112368, -0.0128949, -0.0112445, -0.0007930, 0.0008262
1: -0.0065679, -0.0061067, -0.0065742, -0.0061089, -0.0002236, 0.0002329
2: -0.0098994, -0.0064969, -0.0099462, -0.0065130, -0.0016496, 0.0017186
3: 0.0003173, 0.0007675, 0.0003111, 0.0007654, -0.0002183, 0.0002274
4: 0.0109473, 0.0134900, 0.0109593, 0.0135251, -0.0012844, 0.0012328
5: 0.9985477, 0.9992542, 0.9985511, 0.9992639, -0.0003568, 0.0003425
6: 0.0065654, 0.0072067, 0.0065684, 0.0072155, -0.0003239, 0.0003109
7: 0.0011195, 0.0035125, 0.0011308, 0.0035454, -0.0012088, 0.0011602
8: -0.0119266, -0.0100641, -0.0119523, -0.0100729, -0.0009030, 0.0009408
9: -0.0031415, -0.0029808, -0.0031407, -0.0029786, -0.0000812, 0.0000779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002284, upper bound: 0.0002336
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002450
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128545, -0.0112340, -0.0127982, -0.0111944, -0.0008205, 0.0007185
1: -0.0065628, -0.0061060, -0.0065469, -0.0060948, -0.0002313, 0.0002026
2: -0.0098621, -0.0064912, -0.0097450, -0.0064088, -0.0017069, 0.0014946
3: 0.0003222, 0.0007683, 0.0003377, 0.0007792, -0.0002259, 0.0001978
4: 0.0109430, 0.0134622, 0.0108814, 0.0133747, -0.0011170, 0.0012756
5: 0.9985466, 0.9992465, 0.9985294, 0.9992221, -0.0003103, 0.0003544
6: 0.0065643, 0.0071996, 0.0065488, 0.0071776, -0.0002817, 0.0003217
7: 0.0011154, 0.0034863, 0.0010575, 0.0034039, -0.0010512, 0.0012005
8: -0.0119063, -0.0100610, -0.0118421, -0.0100159, -0.0009343, 0.0008181
9: -0.0031417, -0.0029825, -0.0031456, -0.0029881, -0.0000706, 0.0000806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001985, upper bound: 0.0001955
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001985, upper bound: 0.0002346
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0128650, -0.0112370, -0.0008186, 0.0007545
1: -0.0065672, -0.0061057, -0.0065658, -0.0061068, -0.0002308, 0.0002127
2: -0.0098946, -0.0064896, -0.0098840, -0.0064975, -0.0017028, 0.0015695
3: 0.0003179, 0.0007685, 0.0003193, 0.0007675, -0.0002253, 0.0002077
4: 0.0109418, 0.0134865, 0.0109477, 0.0134785, -0.0011730, 0.0012726
5: 0.9985462, 0.9992533, 0.9985479, 0.9992510, -0.0003259, 0.0003536
6: 0.0065640, 0.0072058, 0.0065655, 0.0072038, -0.0002958, 0.0003209
7: 0.0011143, 0.0035092, 0.0011198, 0.0035017, -0.0011039, 0.0011977
8: -0.0119241, -0.0100601, -0.0119182, -0.0100644, -0.0009321, 0.0008592
9: -0.0031418, -0.0029810, -0.0031414, -0.0029815, -0.0000741, 0.0000804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001981
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0002453
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0128478, -0.0112248, -0.0008492, 0.0007835
1: -0.0065742, -0.0061089, -0.0065609, -0.0061033, -0.0002394, 0.0002209
2: -0.0099462, -0.0065130, -0.0098481, -0.0064719, -0.0017666, 0.0016299
3: 0.0003111, 0.0007654, 0.0003241, 0.0007708, -0.0002338, 0.0002157
4: 0.0109593, 0.0135251, 0.0109286, 0.0134517, -0.0012181, 0.0013202
5: 0.9985511, 0.9992639, 0.9985425, 0.9992435, -0.0003384, 0.0003668
6: 0.0065684, 0.0072155, 0.0065607, 0.0071970, -0.0003072, 0.0003329
7: 0.0011308, 0.0035454, 0.0011019, 0.0034764, -0.0011464, 0.0012425
8: -0.0119523, -0.0100729, -0.0118986, -0.0100505, -0.0009670, 0.0008922
9: -0.0031407, -0.0029786, -0.0031426, -0.0029832, -0.0000770, 0.0000834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002108, upper bound: 0.0002214
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002276, upper bound: 0.0002497
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0128724, -0.0112368, -0.0008262, 0.0007930
1: -0.0065742, -0.0061089, -0.0065679, -0.0061067, -0.0002329, 0.0002236
2: -0.0099462, -0.0065130, -0.0098994, -0.0064969, -0.0017186, 0.0016496
3: 0.0003111, 0.0007654, 0.0003173, 0.0007675, -0.0002274, 0.0002183
4: 0.0109593, 0.0135251, 0.0109473, 0.0134900, -0.0012328, 0.0012844
5: 0.9985511, 0.9992639, 0.9985477, 0.9992542, -0.0003425, 0.0003568
6: 0.0065684, 0.0072155, 0.0065654, 0.0072067, -0.0003109, 0.0003239
7: 0.0011308, 0.0035454, 0.0011195, 0.0035125, -0.0011602, 0.0012088
8: -0.0119523, -0.0100729, -0.0119266, -0.0100641, -0.0009408, 0.0009030
9: -0.0031407, -0.0029786, -0.0031415, -0.0029808, -0.0000779, 0.0000812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002108, upper bound: 0.0002390
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002276, upper bound: 0.0002498
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0128545, -0.0112340, -0.0128187, -0.0112024, -0.0007940, 0.0007240
1: -0.0065628, -0.0061060, -0.0065527, -0.0060970, -0.0002239, 0.0002041
2: -0.0098621, -0.0064912, -0.0097877, -0.0064255, -0.0016517, 0.0015061
3: 0.0003222, 0.0007683, 0.0003320, 0.0007770, -0.0002186, 0.0001993
4: 0.0109430, 0.0134622, 0.0108939, 0.0134066, -0.0011256, 0.0012344
5: 0.9985466, 0.9992465, 0.9985330, 0.9992310, -0.0003127, 0.0003429
6: 0.0065643, 0.0071996, 0.0065519, 0.0071856, -0.0002838, 0.0003113
7: 0.0011154, 0.0034863, 0.0010692, 0.0034340, -0.0010593, 0.0011617
8: -0.0119063, -0.0100610, -0.0118655, -0.0100250, -0.0009041, 0.0008244
9: -0.0031417, -0.0029825, -0.0031448, -0.0029860, -0.0000711, 0.0000780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002010, upper bound: 0.0001959
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002010, upper bound: 0.0002348
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0128701, -0.0112332, -0.0128878, -0.0112447, -0.0007926, 0.0007628
1: -0.0065672, -0.0061057, -0.0065722, -0.0061090, -0.0002235, 0.0002151
2: -0.0098946, -0.0064896, -0.0099315, -0.0065135, -0.0016488, 0.0015869
3: 0.0003179, 0.0007685, 0.0003130, 0.0007653, -0.0002182, 0.0002100
4: 0.0109418, 0.0134865, 0.0109597, 0.0135141, -0.0011859, 0.0012322
5: 0.9985462, 0.9992533, 0.9985511, 0.9992608, -0.0003295, 0.0003423
6: 0.0065640, 0.0072058, 0.0065685, 0.0072127, -0.0002991, 0.0003107
7: 0.0011143, 0.0035092, 0.0011311, 0.0035351, -0.0011161, 0.0011596
8: -0.0119241, -0.0100601, -0.0119442, -0.0100732, -0.0009025, 0.0008687
9: -0.0031418, -0.0029810, -0.0031407, -0.0029792, -0.0000749, 0.0000779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002207, upper bound: 0.0001985
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002207, upper bound: 0.0002453
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0128701, -0.0112332, -0.0008256, 0.0007929
1: -0.0065742, -0.0061089, -0.0065672, -0.0061057, -0.0002328, 0.0002235
2: -0.0099462, -0.0065130, -0.0098946, -0.0064896, -0.0017175, 0.0016494
3: 0.0003111, 0.0007654, 0.0003179, 0.0007685, -0.0002273, 0.0002183
4: 0.0109593, 0.0135251, 0.0109418, 0.0134865, -0.0012326, 0.0012836
5: 0.9985511, 0.9992639, 0.9985462, 0.9992533, -0.0003425, 0.0003566
6: 0.0065684, 0.0072155, 0.0065640, 0.0072058, -0.0003109, 0.0003237
7: 0.0011308, 0.0035454, 0.0011143, 0.0035092, -0.0011601, 0.0012080
8: -0.0119523, -0.0100729, -0.0119241, -0.0100601, -0.0009402, 0.0009029
9: -0.0031407, -0.0029786, -0.0031418, -0.0029810, -0.0000779, 0.0000811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002125, upper bound: 0.0002217
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002497
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0128949, -0.0112445, -0.0128949, -0.0112445, -0.0008010, 0.0008010
1: -0.0065742, -0.0061089, -0.0065742, -0.0061089, -0.0002258, 0.0002258
2: -0.0099462, -0.0065130, -0.0099462, -0.0065130, -0.0016663, 0.0016663
3: 0.0003111, 0.0007654, 0.0003111, 0.0007654, -0.0002205, 0.0002205
4: 0.0109593, 0.0135251, 0.0109593, 0.0135251, -0.0012453, 0.0012453
5: 0.9985511, 0.9992639, 0.9985511, 0.9992639, -0.0003460, 0.0003460
6: 0.0065684, 0.0072155, 0.0065684, 0.0072155, -0.0003140, 0.0003140
7: 0.0011308, 0.0035454, 0.0011308, 0.0035454, -0.0011720, 0.0011720
8: -0.0119523, -0.0100729, -0.0119523, -0.0100729, -0.0009122, 0.0009122
9: -0.0031407, -0.0029786, -0.0031407, -0.0029786, -0.0000787, 0.0000787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002125, upper bound: 0.0002391
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002497
time: 0.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002292
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002305
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002292
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002298, upper bound: 0.0002304
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001814
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002282
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001814
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002324, upper bound: 0.0002282
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002156
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002301, upper bound: 0.0002307
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002157
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002301, upper bound: 0.0002306
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0001840
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0002327
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002040
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002282, upper bound: 0.0002345
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002209
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002282, upper bound: 0.0002346
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0001850
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0002327
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002046
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002281, upper bound: 0.0002345
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002211
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002281, upper bound: 0.0002346
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002286
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002297
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002285
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002423, upper bound: 0.0002297
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002343, upper bound: 0.0001814
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002470, upper bound: 0.0002275
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002343, upper bound: 0.0001814
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002470, upper bound: 0.0002275
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002405, upper bound: 0.0002156
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002297
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002405, upper bound: 0.0002156
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002425, upper bound: 0.0002298
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002312, upper bound: 0.0001839
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002312, upper bound: 0.0002324
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002224, upper bound: 0.0002040
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002342
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002224, upper bound: 0.0002209
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002341
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0001850
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002327, upper bound: 0.0002323
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002243, upper bound: 0.0002046
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002342
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002243, upper bound: 0.0002211
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002403, upper bound: 0.0002343
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002429
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002451
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002429
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002295, upper bound: 0.0002451
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001951
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002403
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001951
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002404
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002333
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002279, upper bound: 0.0002333
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002451
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0001922, upper bound: 0.0001947
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0001922, upper bound: 0.0002340
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0001972
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002162, upper bound: 0.0002453
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002202
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002497
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002076, upper bound: 0.0002387
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002494
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0001953, upper bound: 0.0001952
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0001953, upper bound: 0.0002343
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0001977
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002179, upper bound: 0.0002453
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002206
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002498
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002096, upper bound: 0.0002389
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002498
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002429
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002451
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002429
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002294, upper bound: 0.0002452
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002212, upper bound: 0.0001959
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002404
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002212, upper bound: 0.0001959
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002317, upper bound: 0.0002402
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002284, upper bound: 0.0002336
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002453
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002284, upper bound: 0.0002336
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002297, upper bound: 0.0002450
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0001985, upper bound: 0.0001955
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0001985, upper bound: 0.0002346
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0001981
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002190, upper bound: 0.0002453
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002108, upper bound: 0.0002214
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002276, upper bound: 0.0002497
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002108, upper bound: 0.0002390
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002276, upper bound: 0.0002498
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002010, upper bound: 0.0001959
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002010, upper bound: 0.0002348
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002207, upper bound: 0.0001985
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002207, upper bound: 0.0002453
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002125, upper bound: 0.0002217
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002497
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002125, upper bound: 0.0002391
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.29
Output dim: 5, lower bound: -0.0002275, upper bound: 0.0002497

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127449, -0.0112295, -0.0007286, 0.0007475
1: -0.0065363, -0.0061090, -0.0065319, -0.0061047, -0.0002054, 0.0002107
2: -0.0096667, -0.0065136, -0.0096342, -0.0064817, -0.0015157, 0.0015549
3: 0.0003481, 0.0007653, 0.0003524, 0.0007695, -0.0002006, 0.0002058
4: 0.0109598, 0.0133161, 0.0109359, 0.0132919, -0.0011620, 0.0011327
5: 0.9985512, 0.9992059, 0.9985445, 0.9991992, -0.0003228, 0.0003147
6: 0.0065686, 0.0071628, 0.0065625, 0.0071567, -0.0002930, 0.0002857
7: 0.0011312, 0.0033488, 0.0011088, 0.0033260, -0.0010936, 0.0010660
8: -0.0117993, -0.0100733, -0.0117815, -0.0100558, -0.0008297, 0.0008512
9: -0.0031407, -0.0029918, -0.0031422, -0.0029933, -0.0000734, 0.0000716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002191
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002305
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127681, -0.0112377, -0.0007294, 0.0007823
1: -0.0065363, -0.0061090, -0.0065385, -0.0061070, -0.0002056, 0.0002205
2: -0.0096667, -0.0065136, -0.0096824, -0.0064989, -0.0015172, 0.0016273
3: 0.0003481, 0.0007653, 0.0003460, 0.0007673, -0.0002008, 0.0002153
4: 0.0109598, 0.0133161, 0.0109487, 0.0133279, -0.0012161, 0.0011339
5: 0.9985512, 0.9992059, 0.9985481, 0.9992091, -0.0003379, 0.0003150
6: 0.0065686, 0.0071628, 0.0065658, 0.0071658, -0.0003067, 0.0002860
7: 0.0011312, 0.0033488, 0.0011208, 0.0033599, -0.0011445, 0.0010671
8: -0.0117993, -0.0100733, -0.0118079, -0.0100652, -0.0008305, 0.0008908
9: -0.0031407, -0.0029918, -0.0031414, -0.0029910, -0.0000769, 0.0000717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002162
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002305
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127374, -0.0112297, -0.0127677, -0.0112445, -0.0006924, 0.0007808
1: -0.0065298, -0.0061047, -0.0065384, -0.0061089, -0.0001952, 0.0002201
2: -0.0096186, -0.0064823, -0.0096816, -0.0065131, -0.0014402, 0.0016242
3: 0.0003544, 0.0007695, 0.0003461, 0.0007654, -0.0001906, 0.0002149
4: 0.0109363, 0.0132802, 0.0109593, 0.0133273, -0.0012138, 0.0010763
5: 0.9985447, 0.9991959, 0.9985510, 0.9992089, -0.0003372, 0.0002990
6: 0.0065626, 0.0071537, 0.0065684, 0.0071656, -0.0003061, 0.0002714
7: 0.0011091, 0.0033150, 0.0011308, 0.0033593, -0.0011423, 0.0010130
8: -0.0117729, -0.0100561, -0.0118074, -0.0100730, -0.0007884, 0.0008891
9: -0.0031421, -0.0029940, -0.0031407, -0.0029910, -0.0000767, 0.0000680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002108
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002298
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127374, -0.0112297, -0.0127878, -0.0112553, -0.0006912, 0.0008128
1: -0.0065298, -0.0061047, -0.0065440, -0.0061119, -0.0001949, 0.0002292
2: -0.0096186, -0.0064823, -0.0097235, -0.0065354, -0.0014378, 0.0016909
3: 0.0003544, 0.0007695, 0.0003405, 0.0007624, -0.0001903, 0.0002238
4: 0.0109363, 0.0132802, 0.0109760, 0.0133586, -0.0012636, 0.0010745
5: 0.9985447, 0.9991959, 0.9985557, 0.9992176, -0.0003511, 0.0002985
6: 0.0065626, 0.0071537, 0.0065727, 0.0071735, -0.0003187, 0.0002710
7: 0.0011091, 0.0033150, 0.0011465, 0.0033888, -0.0011892, 0.0010113
8: -0.0117729, -0.0100561, -0.0118304, -0.0100852, -0.0007871, 0.0009256
9: -0.0031421, -0.0029940, -0.0031396, -0.0029891, -0.0000799, 0.0000679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002076
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002282
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127677, -0.0112445, -0.0006970, 0.0007533
1: -0.0065363, -0.0061090, -0.0065384, -0.0061089, -0.0001965, 0.0002124
2: -0.0096667, -0.0065136, -0.0096816, -0.0065131, -0.0014498, 0.0015670
3: 0.0003481, 0.0007653, 0.0003461, 0.0007654, -0.0001919, 0.0002074
4: 0.0109598, 0.0133161, 0.0109593, 0.0133273, -0.0011711, 0.0010835
5: 0.9985512, 0.9992059, 0.9985510, 0.9992089, -0.0003254, 0.0003010
6: 0.0065686, 0.0071628, 0.0065684, 0.0071656, -0.0002953, 0.0002732
7: 0.0011312, 0.0033488, 0.0011308, 0.0033593, -0.0011021, 0.0010197
8: -0.0117993, -0.0100733, -0.0118074, -0.0100730, -0.0007936, 0.0008578
9: -0.0031407, -0.0029918, -0.0031407, -0.0029910, -0.0000740, 0.0000685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002302
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002324
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127878, -0.0112553, -0.0006971, 0.0007877
1: -0.0065363, -0.0061090, -0.0065440, -0.0061119, -0.0001965, 0.0002221
2: -0.0096667, -0.0065136, -0.0097235, -0.0065354, -0.0014501, 0.0016387
3: 0.0003481, 0.0007653, 0.0003405, 0.0007624, -0.0001919, 0.0002169
4: 0.0109598, 0.0133161, 0.0109760, 0.0133586, -0.0012246, 0.0010837
5: 0.9985512, 0.9992059, 0.9985557, 0.9992176, -0.0003402, 0.0003011
6: 0.0065686, 0.0071628, 0.0065727, 0.0071735, -0.0003088, 0.0002733
7: 0.0011312, 0.0033488, 0.0011465, 0.0033888, -0.0011525, 0.0010199
8: -0.0117993, -0.0100733, -0.0118304, -0.0100852, -0.0007938, 0.0008970
9: -0.0031407, -0.0029918, -0.0031396, -0.0029891, -0.0000774, 0.0000685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002284
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002306
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127610, -0.0112380, -0.0127614, -0.0112448, -0.0007246, 0.0007217
1: -0.0065364, -0.0061071, -0.0065366, -0.0061090, -0.0002043, 0.0002035
2: -0.0096675, -0.0064994, -0.0096684, -0.0065135, -0.0015073, 0.0015012
3: 0.0003480, 0.0007672, 0.0003478, 0.0007653, -0.0001995, 0.0001987
4: 0.0109491, 0.0133168, 0.0109597, 0.0133174, -0.0011219, 0.0011264
5: 0.9985483, 0.9992061, 0.9985511, 0.9992062, -0.0003117, 0.0003130
6: 0.0065659, 0.0071630, 0.0065685, 0.0071631, -0.0002829, 0.0002841
7: 0.0011212, 0.0033494, 0.0011311, 0.0033501, -0.0010558, 0.0010601
8: -0.0117997, -0.0100655, -0.0118002, -0.0100732, -0.0008251, 0.0008218
9: -0.0031413, -0.0029917, -0.0031407, -0.0029917, -0.0000709, 0.0000712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002326
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002328
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0127449, -0.0112295, -0.0007591, 0.0007470
1: -0.0065420, -0.0061120, -0.0065319, -0.0061047, -0.0002140, 0.0002106
2: -0.0097088, -0.0065360, -0.0096342, -0.0064817, -0.0015791, 0.0015540
3: 0.0003425, 0.0007624, 0.0003524, 0.0007695, -0.0002090, 0.0002056
4: 0.0109765, 0.0133476, 0.0109359, 0.0132919, -0.0011614, 0.0011801
5: 0.9985558, 0.9992145, 0.9985445, 0.9991992, -0.0003227, 0.0003279
6: 0.0065728, 0.0071707, 0.0065625, 0.0071567, -0.0002929, 0.0002976
7: 0.0011469, 0.0033784, 0.0011088, 0.0033260, -0.0010930, 0.0011106
8: -0.0118223, -0.0100855, -0.0117815, -0.0100558, -0.0008644, 0.0008507
9: -0.0031396, -0.0029898, -0.0031422, -0.0029933, -0.0000734, 0.0000746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002262
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002346
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0127677, -0.0112445, -0.0007288, 0.0007549
1: -0.0065420, -0.0061120, -0.0065384, -0.0061089, -0.0002055, 0.0002128
2: -0.0097088, -0.0065360, -0.0096816, -0.0065131, -0.0015160, 0.0015703
3: 0.0003425, 0.0007624, 0.0003461, 0.0007654, -0.0002006, 0.0002078
4: 0.0109765, 0.0133476, 0.0109593, 0.0133273, -0.0011735, 0.0011330
5: 0.9985558, 0.9992145, 0.9985510, 0.9992089, -0.0003260, 0.0003148
6: 0.0065728, 0.0071707, 0.0065684, 0.0071656, -0.0002959, 0.0002857
7: 0.0011469, 0.0033784, 0.0011308, 0.0033593, -0.0011044, 0.0010663
8: -0.0118223, -0.0100855, -0.0118074, -0.0100730, -0.0008299, 0.0008596
9: -0.0031396, -0.0029898, -0.0031407, -0.0029910, -0.0000742, 0.0000716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002339
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002345
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127610, -0.0112380, -0.0127838, -0.0112555, -0.0007015, 0.0007303
1: -0.0065364, -0.0061071, -0.0065429, -0.0061120, -0.0001978, 0.0002059
2: -0.0096675, -0.0064994, -0.0097151, -0.0065359, -0.0014592, 0.0015193
3: 0.0003480, 0.0007672, 0.0003417, 0.0007624, -0.0001931, 0.0002011
4: 0.0109491, 0.0133168, 0.0109764, 0.0133523, -0.0011354, 0.0010905
5: 0.9985483, 0.9992061, 0.9985558, 0.9992158, -0.0003154, 0.0003030
6: 0.0065659, 0.0071630, 0.0065727, 0.0071719, -0.0002863, 0.0002750
7: 0.0011212, 0.0033494, 0.0011468, 0.0033829, -0.0010685, 0.0010263
8: -0.0117997, -0.0100655, -0.0118258, -0.0100855, -0.0007988, 0.0008316
9: -0.0031413, -0.0029917, -0.0031396, -0.0029895, -0.0000718, 0.0000689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002327
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002326
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0127681, -0.0112377, -0.0007380, 0.0007579
1: -0.0065420, -0.0061120, -0.0065385, -0.0061070, -0.0002081, 0.0002137
2: -0.0097088, -0.0065360, -0.0096824, -0.0064989, -0.0015353, 0.0015765
3: 0.0003425, 0.0007624, 0.0003460, 0.0007673, -0.0002032, 0.0002086
4: 0.0109765, 0.0133476, 0.0109487, 0.0133279, -0.0011782, 0.0011474
5: 0.9985558, 0.9992145, 0.9985481, 0.9992091, -0.0003273, 0.0003188
6: 0.0065728, 0.0071707, 0.0065658, 0.0071658, -0.0002971, 0.0002893
7: 0.0011469, 0.0033784, 0.0011208, 0.0033599, -0.0011088, 0.0010798
8: -0.0118223, -0.0100855, -0.0118079, -0.0100652, -0.0008404, 0.0008630
9: -0.0031396, -0.0029898, -0.0031414, -0.0029910, -0.0000745, 0.0000725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002264
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002344
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0127878, -0.0112553, -0.0007058, 0.0007640
1: -0.0065420, -0.0061120, -0.0065440, -0.0061119, -0.0001990, 0.0002154
2: -0.0097088, -0.0065360, -0.0097235, -0.0065354, -0.0014682, 0.0015894
3: 0.0003425, 0.0007624, 0.0003405, 0.0007624, -0.0001943, 0.0002103
4: 0.0109765, 0.0133476, 0.0109760, 0.0133586, -0.0011878, 0.0010972
5: 0.9985558, 0.9992145, 0.9985557, 0.9992176, -0.0003300, 0.0003048
6: 0.0065728, 0.0071707, 0.0065727, 0.0071735, -0.0002995, 0.0002767
7: 0.0011469, 0.0033784, 0.0011465, 0.0033888, -0.0011178, 0.0010326
8: -0.0118223, -0.0100855, -0.0118304, -0.0100852, -0.0008037, 0.0008700
9: -0.0031396, -0.0029898, -0.0031396, -0.0029891, -0.0000751, 0.0000693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002339
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002347
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127374, -0.0112297, -0.0128478, -0.0112248, -0.0007679, 0.0009256
1: -0.0065298, -0.0061047, -0.0065609, -0.0061033, -0.0002165, 0.0002610
2: -0.0096186, -0.0064823, -0.0098481, -0.0064719, -0.0015974, 0.0019254
3: 0.0003544, 0.0007695, 0.0003241, 0.0007708, -0.0002114, 0.0002548
4: 0.0109363, 0.0132802, 0.0109286, 0.0134517, -0.0014389, 0.0011938
5: 0.9985447, 0.9991959, 0.9985425, 0.9992435, -0.0003998, 0.0003317
6: 0.0065626, 0.0071537, 0.0065607, 0.0071970, -0.0003629, 0.0003011
7: 0.0011091, 0.0033150, 0.0011019, 0.0034764, -0.0013542, 0.0011235
8: -0.0117729, -0.0100561, -0.0118986, -0.0100505, -0.0008744, 0.0010540
9: -0.0031421, -0.0029940, -0.0031426, -0.0029832, -0.0000909, 0.0000754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002107
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002300
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0128478, -0.0112248, -0.0008065, 0.0009246
1: -0.0065363, -0.0061090, -0.0065609, -0.0061033, -0.0002274, 0.0002607
2: -0.0096667, -0.0065136, -0.0098481, -0.0064719, -0.0016776, 0.0019233
3: 0.0003481, 0.0007653, 0.0003241, 0.0007708, -0.0002220, 0.0002545
4: 0.0109598, 0.0133161, 0.0109286, 0.0134517, -0.0014373, 0.0012537
5: 0.9985512, 0.9992059, 0.9985425, 0.9992435, -0.0003993, 0.0003483
6: 0.0065686, 0.0071628, 0.0065607, 0.0071970, -0.0003625, 0.0003162
7: 0.0011312, 0.0033488, 0.0011019, 0.0034764, -0.0013527, 0.0011799
8: -0.0117993, -0.0100733, -0.0118986, -0.0100505, -0.0009183, 0.0010528
9: -0.0031407, -0.0029918, -0.0031426, -0.0029832, -0.0000908, 0.0000792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002191
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002317
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127374, -0.0112297, -0.0128701, -0.0112332, -0.0007641, 0.0009521
1: -0.0065298, -0.0061047, -0.0065672, -0.0061057, -0.0002154, 0.0002684
2: -0.0096186, -0.0064823, -0.0098946, -0.0064896, -0.0015895, 0.0019805
3: 0.0003544, 0.0007695, 0.0003179, 0.0007685, -0.0002103, 0.0002621
4: 0.0109363, 0.0132802, 0.0109418, 0.0134865, -0.0014801, 0.0011879
5: 0.9985447, 0.9991959, 0.9985462, 0.9992533, -0.0004112, 0.0003300
6: 0.0065626, 0.0071537, 0.0065640, 0.0072058, -0.0003733, 0.0002996
7: 0.0011091, 0.0033150, 0.0011143, 0.0035092, -0.0013930, 0.0011179
8: -0.0117729, -0.0100561, -0.0119241, -0.0100601, -0.0008701, 0.0010842
9: -0.0031421, -0.0029940, -0.0031418, -0.0029810, -0.0000935, 0.0000751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002076
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002286
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0128701, -0.0112332, -0.0008026, 0.0009511
1: -0.0065363, -0.0061090, -0.0065672, -0.0061057, -0.0002263, 0.0002681
2: -0.0096667, -0.0065136, -0.0098946, -0.0064896, -0.0016697, 0.0019785
3: 0.0003481, 0.0007653, 0.0003179, 0.0007685, -0.0002210, 0.0002618
4: 0.0109598, 0.0133161, 0.0109418, 0.0134865, -0.0014786, 0.0012478
5: 0.9985512, 0.9992059, 0.9985462, 0.9992533, -0.0004108, 0.0003467
6: 0.0065686, 0.0071628, 0.0065640, 0.0072058, -0.0003729, 0.0003147
7: 0.0011312, 0.0033488, 0.0011143, 0.0035092, -0.0013915, 0.0011743
8: -0.0117993, -0.0100733, -0.0119241, -0.0100601, -0.0009140, 0.0010830
9: -0.0031407, -0.0029918, -0.0031418, -0.0029810, -0.0000934, 0.0000789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002162
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002297
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0126655, -0.0111921, -0.0128579, -0.0112377, -0.0007192, 0.0009619
1: -0.0065095, -0.0060941, -0.0065638, -0.0061070, -0.0002028, 0.0002712
2: -0.0094690, -0.0064040, -0.0098693, -0.0064988, -0.0014961, 0.0020010
3: 0.0003742, 0.0007798, 0.0003213, 0.0007673, -0.0001980, 0.0002648
4: 0.0108778, 0.0131684, 0.0109487, 0.0134675, -0.0014954, 0.0011181
5: 0.9985284, 0.9991648, 0.9985482, 0.9992480, -0.0004155, 0.0003106
6: 0.0065479, 0.0071255, 0.0065658, 0.0072010, -0.0003771, 0.0002820
7: 0.0010541, 0.0032098, 0.0011208, 0.0034913, -0.0014073, 0.0010523
8: -0.0116910, -0.0100132, -0.0119102, -0.0100652, -0.0008190, 0.0010953
9: -0.0031458, -0.0030011, -0.0031414, -0.0029822, -0.0000945, 0.0000707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002094, upper bound: 0.0001781
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002094, upper bound: 0.0001814
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127374, -0.0112297, -0.0128724, -0.0112368, -0.0007662, 0.0009594
1: -0.0065298, -0.0061047, -0.0065679, -0.0061067, -0.0002160, 0.0002705
2: -0.0096186, -0.0064823, -0.0098994, -0.0064969, -0.0015938, 0.0019957
3: 0.0003544, 0.0007695, 0.0003173, 0.0007675, -0.0002109, 0.0002641
4: 0.0109363, 0.0132802, 0.0109473, 0.0134900, -0.0014915, 0.0011911
5: 0.9985447, 0.9991959, 0.9985477, 0.9992542, -0.0004144, 0.0003309
6: 0.0065626, 0.0071537, 0.0065654, 0.0072067, -0.0003761, 0.0003004
7: 0.0011091, 0.0033150, 0.0011195, 0.0035125, -0.0014036, 0.0011209
8: -0.0117729, -0.0100561, -0.0119266, -0.0100641, -0.0008724, 0.0010924
9: -0.0031421, -0.0029940, -0.0031415, -0.0029808, -0.0000943, 0.0000753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002136, upper bound: 0.0002108
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002136, upper bound: 0.0002294
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0126655, -0.0111921, -0.0128800, -0.0112454, -0.0007145, 0.0009885
1: -0.0065095, -0.0060941, -0.0065700, -0.0061092, -0.0002015, 0.0002787
2: -0.0094690, -0.0064040, -0.0099151, -0.0065149, -0.0014864, 0.0020563
3: 0.0003742, 0.0007798, 0.0003152, 0.0007652, -0.0001967, 0.0002721
4: 0.0108778, 0.0131684, 0.0109607, 0.0135018, -0.0015368, 0.0011108
5: 0.9985284, 0.9991648, 0.9985515, 0.9992574, -0.0004270, 0.0003086
6: 0.0065479, 0.0071255, 0.0065688, 0.0072096, -0.0003875, 0.0002801
7: 0.0010541, 0.0032098, 0.0011321, 0.0035236, -0.0014463, 0.0010454
8: -0.0116910, -0.0100132, -0.0119353, -0.0100740, -0.0008136, 0.0011256
9: -0.0031458, -0.0030011, -0.0031406, -0.0029800, -0.0000971, 0.0000702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002168, upper bound: 0.0001781
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002168, upper bound: 0.0001814
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127374, -0.0112297, -0.0128949, -0.0112445, -0.0007596, 0.0009852
1: -0.0065298, -0.0061047, -0.0065742, -0.0061089, -0.0002141, 0.0002778
2: -0.0096186, -0.0064823, -0.0099462, -0.0065130, -0.0015800, 0.0020493
3: 0.0003544, 0.0007695, 0.0003111, 0.0007654, -0.0002091, 0.0002712
4: 0.0109363, 0.0132802, 0.0109593, 0.0135251, -0.0015315, 0.0011808
5: 0.9985447, 0.9991959, 0.9985511, 0.9992639, -0.0004255, 0.0003281
6: 0.0065626, 0.0071537, 0.0065684, 0.0072155, -0.0003862, 0.0002978
7: 0.0011091, 0.0033150, 0.0011308, 0.0035454, -0.0014414, 0.0011113
8: -0.0117729, -0.0100561, -0.0119523, -0.0100729, -0.0008649, 0.0011218
9: -0.0031421, -0.0029940, -0.0031407, -0.0029786, -0.0000968, 0.0000746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002202, upper bound: 0.0002076
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002202, upper bound: 0.0002274
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0126867, -0.0112058, -0.0128579, -0.0112377, -0.0007283, 0.0009340
1: -0.0065155, -0.0060980, -0.0065638, -0.0061070, -0.0002053, 0.0002633
2: -0.0095132, -0.0064325, -0.0098693, -0.0064988, -0.0015150, 0.0019429
3: 0.0003684, 0.0007761, 0.0003213, 0.0007673, -0.0002005, 0.0002571
4: 0.0108991, 0.0132014, 0.0109487, 0.0134675, -0.0014520, 0.0011322
5: 0.9985344, 0.9991740, 0.9985482, 0.9992480, -0.0004034, 0.0003146
6: 0.0065533, 0.0071339, 0.0065658, 0.0072010, -0.0003662, 0.0002855
7: 0.0010741, 0.0032409, 0.0011208, 0.0034913, -0.0013665, 0.0010655
8: -0.0117152, -0.0100289, -0.0119102, -0.0100652, -0.0008293, 0.0010635
9: -0.0031445, -0.0029990, -0.0031414, -0.0029822, -0.0000918, 0.0000715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002161
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002162
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0128724, -0.0112368, -0.0007734, 0.0009306
1: -0.0065363, -0.0061090, -0.0065679, -0.0061067, -0.0002181, 0.0002624
2: -0.0096667, -0.0065136, -0.0098994, -0.0064969, -0.0016088, 0.0019358
3: 0.0003481, 0.0007653, 0.0003173, 0.0007675, -0.0002129, 0.0002562
4: 0.0109598, 0.0133161, 0.0109473, 0.0134900, -0.0014467, 0.0012023
5: 0.9985512, 0.9992059, 0.9985477, 0.9992542, -0.0004019, 0.0003340
6: 0.0065686, 0.0071628, 0.0065654, 0.0072067, -0.0003648, 0.0003032
7: 0.0011312, 0.0033488, 0.0011195, 0.0035125, -0.0013615, 0.0011315
8: -0.0117993, -0.0100733, -0.0119266, -0.0100641, -0.0008807, 0.0010597
9: -0.0031407, -0.0029918, -0.0031415, -0.0029808, -0.0000914, 0.0000760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002300
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002318
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0126867, -0.0112058, -0.0128800, -0.0112454, -0.0007256, 0.0009609
1: -0.0065155, -0.0060980, -0.0065700, -0.0061092, -0.0002046, 0.0002709
2: -0.0095132, -0.0064325, -0.0099151, -0.0065149, -0.0015095, 0.0019988
3: 0.0003684, 0.0007761, 0.0003152, 0.0007652, -0.0001998, 0.0002645
4: 0.0108991, 0.0132014, 0.0109607, 0.0135018, -0.0014938, 0.0011281
5: 0.9985344, 0.9991740, 0.9985515, 0.9992574, -0.0004150, 0.0003134
6: 0.0065533, 0.0071339, 0.0065688, 0.0072096, -0.0003767, 0.0002845
7: 0.0010741, 0.0032409, 0.0011321, 0.0035236, -0.0014058, 0.0010617
8: -0.0117152, -0.0100289, -0.0119353, -0.0100740, -0.0008263, 0.0010942
9: -0.0031445, -0.0029990, -0.0031406, -0.0029800, -0.0000944, 0.0000713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002155
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002156
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0128949, -0.0112445, -0.0007697, 0.0009569
1: -0.0065363, -0.0061090, -0.0065742, -0.0061089, -0.0002170, 0.0002698
2: -0.0096667, -0.0065136, -0.0099462, -0.0065130, -0.0016012, 0.0019906
3: 0.0003481, 0.0007653, 0.0003111, 0.0007654, -0.0002119, 0.0002634
4: 0.0109598, 0.0133161, 0.0109593, 0.0135251, -0.0014876, 0.0011967
5: 0.9985512, 0.9992059, 0.9985511, 0.9992639, -0.0004133, 0.0003325
6: 0.0065686, 0.0071628, 0.0065684, 0.0072155, -0.0003752, 0.0003018
7: 0.0011312, 0.0033488, 0.0011308, 0.0035454, -0.0014000, 0.0011262
8: -0.0117993, -0.0100733, -0.0119523, -0.0100729, -0.0008765, 0.0010897
9: -0.0031407, -0.0029918, -0.0031407, -0.0029786, -0.0000940, 0.0000756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002282
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002298
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0126873, -0.0112014, -0.0128650, -0.0112370, -0.0007523, 0.0009725
1: -0.0065157, -0.0060967, -0.0065658, -0.0061068, -0.0002121, 0.0002742
2: -0.0095143, -0.0064233, -0.0098840, -0.0064975, -0.0015649, 0.0020231
3: 0.0003682, 0.0007773, 0.0003193, 0.0007675, -0.0002071, 0.0002677
4: 0.0108923, 0.0132023, 0.0109477, 0.0134785, -0.0015119, 0.0011695
5: 0.9985325, 0.9991742, 0.9985479, 0.9992510, -0.0004201, 0.0003249
6: 0.0065515, 0.0071341, 0.0065655, 0.0072038, -0.0003813, 0.0002949
7: 0.0010677, 0.0032417, 0.0011198, 0.0035017, -0.0014229, 0.0011006
8: -0.0117159, -0.0100239, -0.0119182, -0.0100644, -0.0008566, 0.0011074
9: -0.0031449, -0.0029989, -0.0031414, -0.0029815, -0.0000955, 0.0000739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0001835
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0001840
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127610, -0.0112380, -0.0128650, -0.0112370, -0.0007984, 0.0009063
1: -0.0065364, -0.0061071, -0.0065658, -0.0061068, -0.0002251, 0.0002555
2: -0.0096675, -0.0064994, -0.0098840, -0.0064975, -0.0016607, 0.0018853
3: 0.0003480, 0.0007672, 0.0003193, 0.0007675, -0.0002198, 0.0002495
4: 0.0109491, 0.0133168, 0.0109477, 0.0134785, -0.0014090, 0.0012411
5: 0.9985483, 0.9992061, 0.9985479, 0.9992510, -0.0003915, 0.0003448
6: 0.0065659, 0.0071630, 0.0065655, 0.0072038, -0.0003553, 0.0003130
7: 0.0011212, 0.0033494, 0.0011198, 0.0035017, -0.0013260, 0.0011680
8: -0.0117997, -0.0100655, -0.0119182, -0.0100644, -0.0009091, 0.0010320
9: -0.0031413, -0.0029917, -0.0031414, -0.0029815, -0.0000890, 0.0000784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0002324
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0002324
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0128478, -0.0112248, -0.0008369, 0.0009241
1: -0.0065420, -0.0061120, -0.0065609, -0.0061033, -0.0002360, 0.0002605
2: -0.0097088, -0.0065360, -0.0098481, -0.0064719, -0.0017410, 0.0019224
3: 0.0003425, 0.0007624, 0.0003241, 0.0007708, -0.0002304, 0.0002544
4: 0.0109765, 0.0133476, 0.0109286, 0.0134517, -0.0014367, 0.0013011
5: 0.9985558, 0.9992145, 0.9985425, 0.9992435, -0.0003991, 0.0003615
6: 0.0065728, 0.0071707, 0.0065607, 0.0071970, -0.0003623, 0.0003281
7: 0.0011469, 0.0033784, 0.0011019, 0.0034764, -0.0013521, 0.0012245
8: -0.0118223, -0.0100855, -0.0118986, -0.0100505, -0.0009530, 0.0010523
9: -0.0031396, -0.0029898, -0.0031426, -0.0029832, -0.0000908, 0.0000822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001951, upper bound: 0.0002262
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001951, upper bound: 0.0002342
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0128724, -0.0112368, -0.0008052, 0.0009321
1: -0.0065420, -0.0061120, -0.0065679, -0.0061067, -0.0002270, 0.0002628
2: -0.0097088, -0.0065360, -0.0098994, -0.0064969, -0.0016750, 0.0019390
3: 0.0003425, 0.0007624, 0.0003173, 0.0007675, -0.0002217, 0.0002566
4: 0.0109765, 0.0133476, 0.0109473, 0.0134900, -0.0014491, 0.0012518
5: 0.9985558, 0.9992145, 0.9985477, 0.9992542, -0.0004026, 0.0003478
6: 0.0065728, 0.0071707, 0.0065654, 0.0072067, -0.0003654, 0.0003157
7: 0.0011469, 0.0033784, 0.0011195, 0.0035125, -0.0013638, 0.0011781
8: -0.0118223, -0.0100855, -0.0119266, -0.0100641, -0.0009169, 0.0010614
9: -0.0031396, -0.0029898, -0.0031415, -0.0029808, -0.0000916, 0.0000791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002286, upper bound: 0.0002338
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002286, upper bound: 0.0002343
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0126873, -0.0112014, -0.0128878, -0.0112447, -0.0007289, 0.0009805
1: -0.0065157, -0.0060967, -0.0065722, -0.0061090, -0.0002055, 0.0002764
2: -0.0095143, -0.0064233, -0.0099315, -0.0065135, -0.0015163, 0.0020397
3: 0.0003682, 0.0007773, 0.0003130, 0.0007653, -0.0002007, 0.0002699
4: 0.0108923, 0.0132023, 0.0109597, 0.0135141, -0.0015243, 0.0011332
5: 0.9985325, 0.9991742, 0.9985511, 0.9992608, -0.0004235, 0.0003148
6: 0.0065515, 0.0071341, 0.0065685, 0.0072127, -0.0003844, 0.0002858
7: 0.0010677, 0.0032417, 0.0011311, 0.0035351, -0.0014345, 0.0010665
8: -0.0117159, -0.0100239, -0.0119442, -0.0100732, -0.0008300, 0.0011165
9: -0.0031449, -0.0029989, -0.0031407, -0.0029792, -0.0000963, 0.0000716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0001846
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0001850
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127610, -0.0112380, -0.0128878, -0.0112447, -0.0007746, 0.0009155
1: -0.0065364, -0.0061071, -0.0065722, -0.0061090, -0.0002184, 0.0002581
2: -0.0096675, -0.0064994, -0.0099315, -0.0065135, -0.0016113, 0.0019044
3: 0.0003480, 0.0007672, 0.0003130, 0.0007653, -0.0002132, 0.0002520
4: 0.0109491, 0.0133168, 0.0109597, 0.0135141, -0.0014232, 0.0012042
5: 0.9985483, 0.9992061, 0.9985511, 0.9992608, -0.0003954, 0.0003345
6: 0.0065659, 0.0071630, 0.0065685, 0.0072127, -0.0003589, 0.0003037
7: 0.0011212, 0.0033494, 0.0011311, 0.0035351, -0.0013394, 0.0011332
8: -0.0117997, -0.0100655, -0.0119442, -0.0100732, -0.0008820, 0.0010425
9: -0.0031413, -0.0029917, -0.0031407, -0.0029792, -0.0000899, 0.0000761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0002324
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0002323
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0128701, -0.0112332, -0.0008161, 0.0009342
1: -0.0065420, -0.0061120, -0.0065672, -0.0061057, -0.0002301, 0.0002634
2: -0.0097088, -0.0065360, -0.0098946, -0.0064896, -0.0016976, 0.0019434
3: 0.0003425, 0.0007624, 0.0003179, 0.0007685, -0.0002247, 0.0002572
4: 0.0109765, 0.0133476, 0.0109418, 0.0134865, -0.0014523, 0.0012687
5: 0.9985558, 0.9992145, 0.9985462, 0.9992533, -0.0004035, 0.0003525
6: 0.0065728, 0.0071707, 0.0065640, 0.0072058, -0.0003663, 0.0003199
7: 0.0011469, 0.0033784, 0.0011143, 0.0035092, -0.0013668, 0.0011940
8: -0.0118223, -0.0100855, -0.0119241, -0.0100601, -0.0009293, 0.0010638
9: -0.0031396, -0.0029898, -0.0031418, -0.0029810, -0.0000918, 0.0000802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001968, upper bound: 0.0002264
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001968, upper bound: 0.0002342
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0128949, -0.0112445, -0.0007826, 0.0009404
1: -0.0065420, -0.0061120, -0.0065742, -0.0061089, -0.0002207, 0.0002651
2: -0.0097088, -0.0065360, -0.0099462, -0.0065130, -0.0016280, 0.0019562
3: 0.0003425, 0.0007624, 0.0003111, 0.0007654, -0.0002154, 0.0002589
4: 0.0109765, 0.0133476, 0.0109593, 0.0135251, -0.0014620, 0.0012167
5: 0.9985558, 0.9992145, 0.9985511, 0.9992639, -0.0004062, 0.0003380
6: 0.0065728, 0.0071707, 0.0065684, 0.0072155, -0.0003687, 0.0003068
7: 0.0011469, 0.0033784, 0.0011308, 0.0035454, -0.0013759, 0.0011450
8: -0.0118223, -0.0100855, -0.0119523, -0.0100729, -0.0008912, 0.0010708
9: -0.0031396, -0.0029898, -0.0031407, -0.0029786, -0.0000924, 0.0000769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002338
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002343
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0127449, -0.0112295, -0.0008771, 0.0008212
1: -0.0065588, -0.0061034, -0.0065319, -0.0061047, -0.0002473, 0.0002315
2: -0.0098322, -0.0064726, -0.0096342, -0.0064817, -0.0018244, 0.0017084
3: 0.0003262, 0.0007708, 0.0003524, 0.0007695, -0.0002414, 0.0002261
4: 0.0109290, 0.0134398, 0.0109359, 0.0132919, -0.0012767, 0.0013635
5: 0.9985427, 0.9992402, 0.9985445, 0.9991992, -0.0003547, 0.0003788
6: 0.0065608, 0.0071940, 0.0065625, 0.0071567, -0.0003220, 0.0003438
7: 0.0011023, 0.0034652, 0.0011088, 0.0033260, -0.0012015, 0.0012832
8: -0.0118899, -0.0100508, -0.0117815, -0.0100558, -0.0009987, 0.0009352
9: -0.0031426, -0.0029839, -0.0031422, -0.0029933, -0.0000807, 0.0000862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002246
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002445
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0127449, -0.0112295, -0.0009097, 0.0008182
1: -0.0065655, -0.0061068, -0.0065319, -0.0061047, -0.0002565, 0.0002307
2: -0.0098819, -0.0064976, -0.0096342, -0.0064817, -0.0018924, 0.0017020
3: 0.0003196, 0.0007674, 0.0003524, 0.0007695, -0.0002504, 0.0002252
4: 0.0109478, 0.0134770, 0.0109359, 0.0132919, -0.0012720, 0.0014143
5: 0.9985479, 0.9992506, 0.9985445, 0.9991992, -0.0003534, 0.0003929
6: 0.0065655, 0.0072034, 0.0065625, 0.0071567, -0.0003208, 0.0003567
7: 0.0011199, 0.0035002, 0.0011088, 0.0033260, -0.0011971, 0.0013310
8: -0.0119171, -0.0100645, -0.0117815, -0.0100558, -0.0010359, 0.0009317
9: -0.0031414, -0.0029816, -0.0031422, -0.0029933, -0.0000804, 0.0000894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002343
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002471
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0127681, -0.0112377, -0.0008778, 0.0008560
1: -0.0065588, -0.0061034, -0.0065385, -0.0061070, -0.0002475, 0.0002413
2: -0.0098322, -0.0064726, -0.0096824, -0.0064989, -0.0018260, 0.0017807
3: 0.0003262, 0.0007708, 0.0003460, 0.0007673, -0.0002416, 0.0002356
4: 0.0109290, 0.0134398, 0.0109487, 0.0133279, -0.0013308, 0.0013647
5: 0.9985427, 0.9992402, 0.9985481, 0.9992091, -0.0003697, 0.0003791
6: 0.0065608, 0.0071940, 0.0065658, 0.0071658, -0.0003356, 0.0003441
7: 0.0011023, 0.0034652, 0.0011208, 0.0033599, -0.0012524, 0.0012843
8: -0.0118899, -0.0100508, -0.0118079, -0.0100652, -0.0009996, 0.0009748
9: -0.0031426, -0.0029839, -0.0031414, -0.0029910, -0.0000841, 0.0000862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002224
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002429
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0127681, -0.0112377, -0.0009105, 0.0008530
1: -0.0065655, -0.0061068, -0.0065385, -0.0061070, -0.0002567, 0.0002405
2: -0.0098819, -0.0064976, -0.0096824, -0.0064989, -0.0018940, 0.0017743
3: 0.0003196, 0.0007674, 0.0003460, 0.0007673, -0.0002506, 0.0002348
4: 0.0109478, 0.0134770, 0.0109487, 0.0133279, -0.0013260, 0.0014155
5: 0.9985479, 0.9992506, 0.9985481, 0.9992091, -0.0003684, 0.0003933
6: 0.0065655, 0.0072034, 0.0065658, 0.0071658, -0.0003344, 0.0003570
7: 0.0011199, 0.0035002, 0.0011208, 0.0033599, -0.0012479, 0.0013321
8: -0.0119171, -0.0100645, -0.0118079, -0.0100652, -0.0010368, 0.0009713
9: -0.0031414, -0.0029816, -0.0031414, -0.0029910, -0.0000838, 0.0000894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002311
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002451
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0127677, -0.0112445, -0.0008793, 0.0008535
1: -0.0065588, -0.0061034, -0.0065384, -0.0061089, -0.0002479, 0.0002406
2: -0.0098322, -0.0064726, -0.0096816, -0.0065131, -0.0018292, 0.0017756
3: 0.0003262, 0.0007708, 0.0003461, 0.0007654, -0.0002421, 0.0002350
4: 0.0109290, 0.0134398, 0.0109593, 0.0133273, -0.0013269, 0.0013670
5: 0.9985427, 0.9992402, 0.9985510, 0.9992089, -0.0003687, 0.0003798
6: 0.0065608, 0.0071940, 0.0065684, 0.0071656, -0.0003346, 0.0003447
7: 0.0011023, 0.0034652, 0.0011308, 0.0033593, -0.0012488, 0.0012865
8: -0.0118899, -0.0100508, -0.0118074, -0.0100730, -0.0010013, 0.0009719
9: -0.0031426, -0.0029839, -0.0031407, -0.0029910, -0.0000839, 0.0000864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002246
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002423
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0127878, -0.0112553, -0.0008782, 0.0008856
1: -0.0065588, -0.0061034, -0.0065440, -0.0061119, -0.0002476, 0.0002497
2: -0.0098322, -0.0064726, -0.0097235, -0.0065354, -0.0018268, 0.0018422
3: 0.0003262, 0.0007708, 0.0003405, 0.0007624, -0.0002417, 0.0002438
4: 0.0109290, 0.0134398, 0.0109760, 0.0133586, -0.0013768, 0.0013652
5: 0.9985427, 0.9992402, 0.9985557, 0.9992176, -0.0003825, 0.0003793
6: 0.0065608, 0.0071940, 0.0065727, 0.0071735, -0.0003472, 0.0003443
7: 0.0011023, 0.0034652, 0.0011465, 0.0033888, -0.0012957, 0.0012848
8: -0.0118899, -0.0100508, -0.0118304, -0.0100852, -0.0010000, 0.0010084
9: -0.0031426, -0.0029839, -0.0031396, -0.0029891, -0.0000870, 0.0000863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002224
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002404
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127987, -0.0111945, -0.0127516, -0.0112453, -0.0008414, 0.0008308
1: -0.0065471, -0.0060948, -0.0065338, -0.0061091, -0.0002372, 0.0002342
2: -0.0097460, -0.0064089, -0.0096482, -0.0065147, -0.0017503, 0.0017282
3: 0.0003376, 0.0007792, 0.0003505, 0.0007652, -0.0002316, 0.0002287
4: 0.0108815, 0.0133754, 0.0109605, 0.0133023, -0.0012915, 0.0013081
5: 0.9985294, 0.9992223, 0.9985514, 0.9992020, -0.0003588, 0.0003634
6: 0.0065488, 0.0071777, 0.0065688, 0.0071593, -0.0003257, 0.0003299
7: 0.0010576, 0.0034046, 0.0011319, 0.0033358, -0.0012155, 0.0012310
8: -0.0118427, -0.0100160, -0.0117891, -0.0100739, -0.0009581, 0.0009460
9: -0.0031456, -0.0029880, -0.0031406, -0.0029926, -0.0000816, 0.0000827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002337
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002338
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0127677, -0.0112445, -0.0008833, 0.0008265
1: -0.0065655, -0.0061068, -0.0065384, -0.0061089, -0.0002490, 0.0002330
2: -0.0098819, -0.0064976, -0.0096816, -0.0065131, -0.0018374, 0.0017192
3: 0.0003196, 0.0007674, 0.0003461, 0.0007654, -0.0002432, 0.0002275
4: 0.0109478, 0.0134770, 0.0109593, 0.0133273, -0.0012848, 0.0013732
5: 0.9985479, 0.9992506, 0.9985510, 0.9992089, -0.0003570, 0.0003815
6: 0.0065655, 0.0072034, 0.0065684, 0.0071656, -0.0003240, 0.0003463
7: 0.0011199, 0.0035002, 0.0011308, 0.0033593, -0.0012092, 0.0012923
8: -0.0119171, -0.0100645, -0.0118074, -0.0100730, -0.0010058, 0.0009411
9: -0.0031414, -0.0029816, -0.0031407, -0.0029910, -0.0000812, 0.0000868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002445
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002471
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127987, -0.0111945, -0.0127719, -0.0112561, -0.0008429, 0.0008646
1: -0.0065471, -0.0060948, -0.0065395, -0.0061122, -0.0002377, 0.0002438
2: -0.0097460, -0.0064089, -0.0096902, -0.0065371, -0.0017535, 0.0017986
3: 0.0003376, 0.0007792, 0.0003449, 0.0007622, -0.0002320, 0.0002380
4: 0.0108815, 0.0133754, 0.0109773, 0.0133338, -0.0013441, 0.0013104
5: 0.9985294, 0.9992223, 0.9985561, 0.9992108, -0.0003734, 0.0003641
6: 0.0065488, 0.0071777, 0.0065730, 0.0071672, -0.0003390, 0.0003305
7: 0.0010576, 0.0034046, 0.0011477, 0.0033654, -0.0012650, 0.0012333
8: -0.0118427, -0.0100160, -0.0118122, -0.0100861, -0.0009599, 0.0009845
9: -0.0031456, -0.0029880, -0.0031396, -0.0029906, -0.0000849, 0.0000828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002330
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002333
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0127878, -0.0112553, -0.0008834, 0.0008609
1: -0.0065655, -0.0061068, -0.0065440, -0.0061119, -0.0002491, 0.0002427
2: -0.0098819, -0.0064976, -0.0097235, -0.0065354, -0.0018376, 0.0017908
3: 0.0003196, 0.0007674, 0.0003405, 0.0007624, -0.0002432, 0.0002370
4: 0.0109478, 0.0134770, 0.0109760, 0.0133586, -0.0013384, 0.0013733
5: 0.9985479, 0.9992506, 0.9985557, 0.9992176, -0.0003718, 0.0003816
6: 0.0065655, 0.0072034, 0.0065727, 0.0071735, -0.0003375, 0.0003463
7: 0.0011199, 0.0035002, 0.0011465, 0.0033888, -0.0012596, 0.0012925
8: -0.0119171, -0.0100645, -0.0118304, -0.0100852, -0.0010059, 0.0009803
9: -0.0031414, -0.0029816, -0.0031396, -0.0029891, -0.0000846, 0.0000868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002426
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002452
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0126882, -0.0112058, -0.0009705, 0.0007523
1: -0.0065651, -0.0061058, -0.0065159, -0.0060980, -0.0002736, 0.0002121
2: -0.0098789, -0.0064902, -0.0095163, -0.0064324, -0.0020188, 0.0015650
3: 0.0003200, 0.0007684, 0.0003680, 0.0007761, -0.0002672, 0.0002071
4: 0.0109422, 0.0134748, 0.0108990, 0.0132037, -0.0011696, 0.0015087
5: 0.9985464, 0.9992499, 0.9985343, 0.9991746, -0.0003250, 0.0004192
6: 0.0065641, 0.0072028, 0.0065532, 0.0071345, -0.0002950, 0.0003805
7: 0.0011147, 0.0034981, 0.0010741, 0.0032431, -0.0011007, 0.0014199
8: -0.0119155, -0.0100604, -0.0117169, -0.0100288, -0.0011051, 0.0008567
9: -0.0031418, -0.0029817, -0.0031445, -0.0029989, -0.0000739, 0.0000953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002340
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002340
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0127614, -0.0112448, -0.0009051, 0.0007949
1: -0.0065651, -0.0061058, -0.0065366, -0.0061090, -0.0002552, 0.0002241
2: -0.0098789, -0.0064902, -0.0096684, -0.0065135, -0.0018827, 0.0016536
3: 0.0003200, 0.0007684, 0.0003478, 0.0007653, -0.0002491, 0.0002188
4: 0.0109422, 0.0134748, 0.0109597, 0.0133174, -0.0012358, 0.0014070
5: 0.9985464, 0.9992499, 0.9985511, 0.9992062, -0.0003433, 0.0003909
6: 0.0065641, 0.0072028, 0.0065685, 0.0071631, -0.0003116, 0.0003548
7: 0.0011147, 0.0034981, 0.0011311, 0.0033501, -0.0011630, 0.0013241
8: -0.0119155, -0.0100604, -0.0118002, -0.0100732, -0.0010306, 0.0009052
9: -0.0031418, -0.0029817, -0.0031407, -0.0029917, -0.0000781, 0.0000889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002453
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002453
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0127449, -0.0112295, -0.0009359, 0.0008135
1: -0.0065720, -0.0061090, -0.0065319, -0.0061047, -0.0002639, 0.0002294
2: -0.0099299, -0.0065137, -0.0096342, -0.0064817, -0.0019468, 0.0016923
3: 0.0003132, 0.0007653, 0.0003524, 0.0007695, -0.0002576, 0.0002239
4: 0.0109598, 0.0135129, 0.0109359, 0.0132919, -0.0012647, 0.0014549
5: 0.9985512, 0.9992605, 0.9985445, 0.9991992, -0.0003514, 0.0004042
6: 0.0065686, 0.0072124, 0.0065625, 0.0071567, -0.0003189, 0.0003669
7: 0.0011312, 0.0035340, 0.0011088, 0.0033260, -0.0011902, 0.0013692
8: -0.0119434, -0.0100733, -0.0117815, -0.0100558, -0.0010657, 0.0009264
9: -0.0031407, -0.0029793, -0.0031422, -0.0029933, -0.0000799, 0.0000919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002418
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002497
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128198, -0.0112025, -0.0127516, -0.0112453, -0.0008674, 0.0008272
1: -0.0065530, -0.0060971, -0.0065338, -0.0061091, -0.0002445, 0.0002332
2: -0.0097900, -0.0064256, -0.0096482, -0.0065147, -0.0018043, 0.0017208
3: 0.0003317, 0.0007770, 0.0003505, 0.0007652, -0.0002388, 0.0002277
4: 0.0108940, 0.0134083, 0.0109605, 0.0133023, -0.0012860, 0.0013484
5: 0.9985330, 0.9992315, 0.9985514, 0.9992020, -0.0003573, 0.0003746
6: 0.0065520, 0.0071860, 0.0065688, 0.0071593, -0.0003243, 0.0003400
7: 0.0010693, 0.0034356, 0.0011319, 0.0033358, -0.0012103, 0.0012690
8: -0.0118668, -0.0100251, -0.0117891, -0.0100739, -0.0009877, 0.0009420
9: -0.0031448, -0.0029859, -0.0031406, -0.0029926, -0.0000813, 0.0000852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002133, upper bound: 0.0002388
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002133, upper bound: 0.0002388
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0127677, -0.0112445, -0.0009092, 0.0008238
1: -0.0065720, -0.0061090, -0.0065384, -0.0061089, -0.0002563, 0.0002323
2: -0.0099299, -0.0065137, -0.0096816, -0.0065131, -0.0018914, 0.0017137
3: 0.0003132, 0.0007653, 0.0003461, 0.0007654, -0.0002503, 0.0002268
4: 0.0109598, 0.0135129, 0.0109593, 0.0133273, -0.0012807, 0.0014135
5: 0.9985512, 0.9992605, 0.9985510, 0.9992089, -0.0003558, 0.0003927
6: 0.0065686, 0.0072124, 0.0065684, 0.0071656, -0.0003230, 0.0003565
7: 0.0011312, 0.0035340, 0.0011308, 0.0033593, -0.0012053, 0.0013303
8: -0.0119434, -0.0100733, -0.0118074, -0.0100730, -0.0010354, 0.0009381
9: -0.0031407, -0.0029793, -0.0031407, -0.0029910, -0.0000809, 0.0000893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002487
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002498
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0127102, -0.0112166, -0.0009536, 0.0007639
1: -0.0065651, -0.0061058, -0.0065221, -0.0061010, -0.0002689, 0.0002154
2: -0.0098789, -0.0064902, -0.0095619, -0.0064550, -0.0019838, 0.0015890
3: 0.0003200, 0.0007684, 0.0003619, 0.0007731, -0.0002625, 0.0002103
4: 0.0109422, 0.0134748, 0.0109159, 0.0132378, -0.0011875, 0.0014825
5: 0.9985464, 0.9992499, 0.9985390, 0.9991841, -0.0003299, 0.0004119
6: 0.0065641, 0.0072028, 0.0065575, 0.0071431, -0.0002995, 0.0003739
7: 0.0011147, 0.0034981, 0.0010900, 0.0032751, -0.0011176, 0.0013952
8: -0.0119155, -0.0100604, -0.0117419, -0.0100412, -0.0010859, 0.0008698
9: -0.0031418, -0.0029817, -0.0031434, -0.0029967, -0.0000750, 0.0000937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002343
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002343
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0127838, -0.0112555, -0.0008887, 0.0008084
1: -0.0065651, -0.0061058, -0.0065429, -0.0061120, -0.0002506, 0.0002279
2: -0.0098789, -0.0064902, -0.0097151, -0.0065359, -0.0018486, 0.0016816
3: 0.0003200, 0.0007684, 0.0003417, 0.0007624, -0.0002446, 0.0002225
4: 0.0109422, 0.0134748, 0.0109764, 0.0133523, -0.0012567, 0.0013816
5: 0.9985464, 0.9992499, 0.9985558, 0.9992158, -0.0003492, 0.0003838
6: 0.0065641, 0.0072028, 0.0065727, 0.0071719, -0.0003169, 0.0003484
7: 0.0011147, 0.0034981, 0.0011468, 0.0033829, -0.0011827, 0.0013002
8: -0.0119155, -0.0100604, -0.0118258, -0.0100855, -0.0010119, 0.0009205
9: -0.0031418, -0.0029817, -0.0031396, -0.0029895, -0.0000794, 0.0000873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002453
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002454
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0127681, -0.0112377, -0.0009192, 0.0008287
1: -0.0065720, -0.0061090, -0.0065385, -0.0061070, -0.0002592, 0.0002336
2: -0.0099299, -0.0065137, -0.0096824, -0.0064989, -0.0019121, 0.0017238
3: 0.0003132, 0.0007653, 0.0003460, 0.0007673, -0.0002530, 0.0002281
4: 0.0109598, 0.0135129, 0.0109487, 0.0133279, -0.0012882, 0.0014290
5: 0.9985512, 0.9992605, 0.9985481, 0.9992091, -0.0003579, 0.0003970
6: 0.0065686, 0.0072124, 0.0065658, 0.0071658, -0.0003249, 0.0003604
7: 0.0011312, 0.0035340, 0.0011208, 0.0033599, -0.0012124, 0.0013449
8: -0.0119434, -0.0100733, -0.0118079, -0.0100652, -0.0010467, 0.0009436
9: -0.0031407, -0.0029793, -0.0031414, -0.0029910, -0.0000814, 0.0000903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002421
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002497
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128198, -0.0112025, -0.0127719, -0.0112561, -0.0008511, 0.0008406
1: -0.0065530, -0.0060971, -0.0065395, -0.0061122, -0.0002399, 0.0002370
2: -0.0097900, -0.0064256, -0.0096902, -0.0065371, -0.0017704, 0.0017486
3: 0.0003317, 0.0007770, 0.0003449, 0.0007622, -0.0002343, 0.0002314
4: 0.0108940, 0.0134083, 0.0109773, 0.0133338, -0.0013068, 0.0013231
5: 0.9985330, 0.9992315, 0.9985561, 0.9992108, -0.0003631, 0.0003676
6: 0.0065520, 0.0071860, 0.0065730, 0.0071672, -0.0003296, 0.0003337
7: 0.0010693, 0.0034356, 0.0011477, 0.0033654, -0.0012298, 0.0012452
8: -0.0118668, -0.0100251, -0.0118122, -0.0100861, -0.0009691, 0.0009572
9: -0.0031448, -0.0029859, -0.0031396, -0.0029906, -0.0000826, 0.0000836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002153, upper bound: 0.0002389
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002153, upper bound: 0.0002388
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0127878, -0.0112553, -0.0008923, 0.0008367
1: -0.0065720, -0.0061090, -0.0065440, -0.0061119, -0.0002516, 0.0002359
2: -0.0099299, -0.0065137, -0.0097235, -0.0065354, -0.0018561, 0.0017404
3: 0.0003132, 0.0007653, 0.0003405, 0.0007624, -0.0002456, 0.0002303
4: 0.0109598, 0.0135129, 0.0109760, 0.0133586, -0.0013007, 0.0013871
5: 0.9985512, 0.9992605, 0.9985557, 0.9992176, -0.0003614, 0.0003854
6: 0.0065686, 0.0072124, 0.0065727, 0.0071735, -0.0003280, 0.0003498
7: 0.0011312, 0.0035340, 0.0011465, 0.0033888, -0.0012241, 0.0013054
8: -0.0119434, -0.0100733, -0.0118304, -0.0100852, -0.0010160, 0.0009527
9: -0.0031407, -0.0029793, -0.0031396, -0.0029891, -0.0000822, 0.0000877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002487
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002497
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0128478, -0.0112248, -0.0007241, 0.0007843
1: -0.0065588, -0.0061034, -0.0065609, -0.0061033, -0.0002041, 0.0002211
2: -0.0098322, -0.0064726, -0.0098481, -0.0064719, -0.0015062, 0.0016315
3: 0.0003262, 0.0007708, 0.0003241, 0.0007708, -0.0001993, 0.0002159
4: 0.0109290, 0.0134398, 0.0109286, 0.0134517, -0.0012193, 0.0011256
5: 0.9985427, 0.9992402, 0.9985425, 0.9992435, -0.0003388, 0.0003127
6: 0.0065608, 0.0071940, 0.0065607, 0.0071970, -0.0003075, 0.0002839
7: 0.0011023, 0.0034652, 0.0011019, 0.0034764, -0.0011475, 0.0010593
8: -0.0118899, -0.0100508, -0.0118986, -0.0100505, -0.0008245, 0.0008931
9: -0.0031426, -0.0029839, -0.0031426, -0.0029832, -0.0000771, 0.0000711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002251
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002446
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0128478, -0.0112248, -0.0007630, 0.0007839
1: -0.0065655, -0.0061068, -0.0065609, -0.0061033, -0.0002151, 0.0002210
2: -0.0098819, -0.0064976, -0.0098481, -0.0064719, -0.0015872, 0.0016306
3: 0.0003196, 0.0007674, 0.0003241, 0.0007708, -0.0002100, 0.0002158
4: 0.0109478, 0.0134770, 0.0109286, 0.0134517, -0.0012186, 0.0011861
5: 0.9985479, 0.9992506, 0.9985425, 0.9992435, -0.0003386, 0.0003295
6: 0.0065655, 0.0072034, 0.0065607, 0.0071970, -0.0003073, 0.0002991
7: 0.0011199, 0.0035002, 0.0011019, 0.0034764, -0.0011468, 0.0011163
8: -0.0119171, -0.0100645, -0.0118986, -0.0100505, -0.0008688, 0.0008926
9: -0.0031414, -0.0029816, -0.0031426, -0.0029832, -0.0000770, 0.0000750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002347
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002471
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0128701, -0.0112332, -0.0007235, 0.0008190
1: -0.0065588, -0.0061034, -0.0065672, -0.0061057, -0.0002040, 0.0002309
2: -0.0098322, -0.0064726, -0.0098946, -0.0064896, -0.0015050, 0.0017036
3: 0.0003262, 0.0007708, 0.0003179, 0.0007685, -0.0001992, 0.0002254
4: 0.0109290, 0.0134398, 0.0109418, 0.0134865, -0.0012732, 0.0011247
5: 0.9985427, 0.9992402, 0.9985462, 0.9992533, -0.0003537, 0.0003125
6: 0.0065608, 0.0071940, 0.0065640, 0.0072058, -0.0003211, 0.0002836
7: 0.0011023, 0.0034652, 0.0011143, 0.0035092, -0.0011982, 0.0010585
8: -0.0118899, -0.0100508, -0.0119241, -0.0100601, -0.0008238, 0.0009326
9: -0.0031426, -0.0029839, -0.0031418, -0.0029810, -0.0000805, 0.0000711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002230
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002430
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0128701, -0.0112332, -0.0007624, 0.0008185
1: -0.0065655, -0.0061068, -0.0065672, -0.0061057, -0.0002149, 0.0002308
2: -0.0098819, -0.0064976, -0.0098946, -0.0064896, -0.0015859, 0.0017027
3: 0.0003196, 0.0007674, 0.0003179, 0.0007685, -0.0002099, 0.0002253
4: 0.0109478, 0.0134770, 0.0109418, 0.0134865, -0.0012725, 0.0011852
5: 0.9985479, 0.9992506, 0.9985462, 0.9992533, -0.0003535, 0.0003293
6: 0.0065655, 0.0072034, 0.0065640, 0.0072058, -0.0003209, 0.0002989
7: 0.0011199, 0.0035002, 0.0011143, 0.0035092, -0.0011976, 0.0011154
8: -0.0119171, -0.0100645, -0.0119241, -0.0100601, -0.0008681, 0.0009321
9: -0.0031414, -0.0029816, -0.0031418, -0.0029810, -0.0000804, 0.0000749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002316
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002452
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0128724, -0.0112368, -0.0007269, 0.0008170
1: -0.0065588, -0.0061034, -0.0065679, -0.0061067, -0.0002049, 0.0002303
2: -0.0098322, -0.0064726, -0.0098994, -0.0064969, -0.0015122, 0.0016996
3: 0.0003262, 0.0007708, 0.0003173, 0.0007675, -0.0002001, 0.0002249
4: 0.0109290, 0.0134398, 0.0109473, 0.0134900, -0.0012701, 0.0011301
5: 0.9985427, 0.9992402, 0.9985477, 0.9992542, -0.0003529, 0.0003140
6: 0.0065608, 0.0071940, 0.0065654, 0.0072067, -0.0003203, 0.0002850
7: 0.0011023, 0.0034652, 0.0011195, 0.0035125, -0.0011953, 0.0010635
8: -0.0118899, -0.0100508, -0.0119266, -0.0100641, -0.0008278, 0.0009303
9: -0.0031426, -0.0029839, -0.0031415, -0.0029808, -0.0000803, 0.0000714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002022, upper bound: 0.0002251
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002022, upper bound: 0.0002422
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128401, -0.0112251, -0.0128949, -0.0112445, -0.0007237, 0.0008489
1: -0.0065588, -0.0061034, -0.0065742, -0.0061089, -0.0002040, 0.0002393
2: -0.0098322, -0.0064726, -0.0099462, -0.0065130, -0.0015055, 0.0017658
3: 0.0003262, 0.0007708, 0.0003111, 0.0007654, -0.0001992, 0.0002337
4: 0.0109290, 0.0134398, 0.0109593, 0.0135251, -0.0013197, 0.0011251
5: 0.9985427, 0.9992402, 0.9985511, 0.9992639, -0.0003666, 0.0003126
6: 0.0065608, 0.0071940, 0.0065684, 0.0072155, -0.0003328, 0.0002837
7: 0.0011023, 0.0034652, 0.0011308, 0.0035454, -0.0012420, 0.0010588
8: -0.0118899, -0.0100508, -0.0119523, -0.0100729, -0.0008241, 0.0009666
9: -0.0031426, -0.0029839, -0.0031407, -0.0029786, -0.0000834, 0.0000711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002087, upper bound: 0.0002231
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002087, upper bound: 0.0002403
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0127987, -0.0111945, -0.0128579, -0.0112377, -0.0006938, 0.0007918
1: -0.0065471, -0.0060948, -0.0065638, -0.0061070, -0.0001956, 0.0002232
2: -0.0097460, -0.0064089, -0.0098693, -0.0064988, -0.0014433, 0.0016471
3: 0.0003376, 0.0007792, 0.0003213, 0.0007673, -0.0001910, 0.0002180
4: 0.0108815, 0.0133754, 0.0109487, 0.0134675, -0.0012310, 0.0010787
5: 0.9985294, 0.9992223, 0.9985482, 0.9992480, -0.0003420, 0.0002997
6: 0.0065488, 0.0071777, 0.0065658, 0.0072010, -0.0003104, 0.0002720
7: 0.0010576, 0.0034046, 0.0011208, 0.0034913, -0.0011585, 0.0010151
8: -0.0118427, -0.0100160, -0.0119102, -0.0100652, -0.0007901, 0.0009016
9: -0.0031456, -0.0029880, -0.0031414, -0.0029822, -0.0000778, 0.0000682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002170, upper bound: 0.0002339
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002170, upper bound: 0.0002341
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0128724, -0.0112368, -0.0007314, 0.0007918
1: -0.0065655, -0.0061068, -0.0065679, -0.0061067, -0.0002062, 0.0002232
2: -0.0098819, -0.0064976, -0.0098994, -0.0064969, -0.0015214, 0.0016471
3: 0.0003196, 0.0007674, 0.0003173, 0.0007675, -0.0002013, 0.0002180
4: 0.0109478, 0.0134770, 0.0109473, 0.0134900, -0.0012310, 0.0011370
5: 0.9985479, 0.9992506, 0.9985477, 0.9992542, -0.0003420, 0.0003159
6: 0.0065655, 0.0072034, 0.0065654, 0.0072067, -0.0003104, 0.0002867
7: 0.0011199, 0.0035002, 0.0011195, 0.0035125, -0.0011585, 0.0010700
8: -0.0119171, -0.0100645, -0.0119266, -0.0100641, -0.0008328, 0.0009016
9: -0.0031414, -0.0029816, -0.0031415, -0.0029808, -0.0000778, 0.0000719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002171, upper bound: 0.0002444
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002171, upper bound: 0.0002471
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0127987, -0.0111945, -0.0128800, -0.0112454, -0.0006948, 0.0008258
1: -0.0065471, -0.0060948, -0.0065700, -0.0061092, -0.0001959, 0.0002328
2: -0.0097460, -0.0064089, -0.0099151, -0.0065149, -0.0014453, 0.0017178
3: 0.0003376, 0.0007792, 0.0003152, 0.0007652, -0.0001913, 0.0002273
4: 0.0108815, 0.0133754, 0.0109607, 0.0135018, -0.0012837, 0.0010801
5: 0.9985294, 0.9992223, 0.9985515, 0.9992574, -0.0003567, 0.0003001
6: 0.0065488, 0.0071777, 0.0065688, 0.0072096, -0.0003237, 0.0002724
7: 0.0010576, 0.0034046, 0.0011321, 0.0035236, -0.0012081, 0.0010165
8: -0.0118427, -0.0100160, -0.0119353, -0.0100740, -0.0007911, 0.0009403
9: -0.0031456, -0.0029880, -0.0031406, -0.0029800, -0.0000811, 0.0000683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002334
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002336
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128640, -0.0112371, -0.0128949, -0.0112445, -0.0007298, 0.0008258
1: -0.0065655, -0.0061068, -0.0065742, -0.0061089, -0.0002057, 0.0002328
2: -0.0098819, -0.0064976, -0.0099462, -0.0065130, -0.0015181, 0.0017179
3: 0.0003196, 0.0007674, 0.0003111, 0.0007654, -0.0002009, 0.0002273
4: 0.0109478, 0.0134770, 0.0109593, 0.0135251, -0.0012838, 0.0011345
5: 0.9985479, 0.9992506, 0.9985511, 0.9992639, -0.0003567, 0.0003152
6: 0.0065655, 0.0072034, 0.0065684, 0.0072155, -0.0003238, 0.0002861
7: 0.0011199, 0.0035002, 0.0011308, 0.0035454, -0.0012082, 0.0010677
8: -0.0119171, -0.0100645, -0.0119523, -0.0100729, -0.0008310, 0.0009404
9: -0.0031414, -0.0029816, -0.0031407, -0.0029786, -0.0000811, 0.0000717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002426
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002452
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0127982, -0.0111944, -0.0008400, 0.0007190
1: -0.0065651, -0.0061058, -0.0065469, -0.0060948, -0.0002368, 0.0002027
2: -0.0098789, -0.0064902, -0.0097450, -0.0064088, -0.0017474, 0.0014958
3: 0.0003200, 0.0007684, 0.0003377, 0.0007792, -0.0002312, 0.0001979
4: 0.0109422, 0.0134748, 0.0108814, 0.0133747, -0.0011178, 0.0013059
5: 0.9985464, 0.9992499, 0.9985294, 0.9992221, -0.0003106, 0.0003628
6: 0.0065641, 0.0072028, 0.0065488, 0.0071776, -0.0002819, 0.0003293
7: 0.0011147, 0.0034981, 0.0010575, 0.0034039, -0.0010520, 0.0012290
8: -0.0119155, -0.0100604, -0.0118421, -0.0100159, -0.0009565, 0.0008188
9: -0.0031418, -0.0029817, -0.0031456, -0.0029881, -0.0000706, 0.0000825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001828, upper bound: 0.0002346
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001828, upper bound: 0.0002346
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0128650, -0.0112370, -0.0007582, 0.0007541
1: -0.0065651, -0.0061058, -0.0065658, -0.0061068, -0.0002138, 0.0002126
2: -0.0098789, -0.0064902, -0.0098840, -0.0064975, -0.0015772, 0.0015687
3: 0.0003200, 0.0007684, 0.0003193, 0.0007675, -0.0002087, 0.0002076
4: 0.0109422, 0.0134748, 0.0109477, 0.0134785, -0.0011724, 0.0011787
5: 0.9985464, 0.9992499, 0.9985479, 0.9992510, -0.0003257, 0.0003275
6: 0.0065641, 0.0072028, 0.0065655, 0.0072038, -0.0002957, 0.0002973
7: 0.0011147, 0.0034981, 0.0011198, 0.0035017, -0.0011033, 0.0011093
8: -0.0119155, -0.0100604, -0.0119182, -0.0100644, -0.0008634, 0.0008587
9: -0.0031418, -0.0029817, -0.0031414, -0.0029815, -0.0000741, 0.0000745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001827, upper bound: 0.0002454
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001827, upper bound: 0.0002453
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0128478, -0.0112248, -0.0007918, 0.0007832
1: -0.0065720, -0.0061090, -0.0065609, -0.0061033, -0.0002232, 0.0002208
2: -0.0099299, -0.0065137, -0.0098481, -0.0064719, -0.0016471, 0.0016292
3: 0.0003132, 0.0007653, 0.0003241, 0.0007708, -0.0002180, 0.0002156
4: 0.0109598, 0.0135129, 0.0109286, 0.0134517, -0.0012175, 0.0012310
5: 0.9985512, 0.9992605, 0.9985425, 0.9992435, -0.0003383, 0.0003420
6: 0.0065686, 0.0072124, 0.0065607, 0.0071970, -0.0003070, 0.0003104
7: 0.0011312, 0.0035340, 0.0011019, 0.0034764, -0.0011458, 0.0011585
8: -0.0119434, -0.0100733, -0.0118986, -0.0100505, -0.0009016, 0.0008918
9: -0.0031407, -0.0029793, -0.0031426, -0.0029832, -0.0000769, 0.0000778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001870, upper bound: 0.0002421
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001870, upper bound: 0.0002496
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128198, -0.0112025, -0.0128579, -0.0112377, -0.0007245, 0.0007899
1: -0.0065530, -0.0060971, -0.0065638, -0.0061070, -0.0002043, 0.0002227
2: -0.0097900, -0.0064256, -0.0098693, -0.0064988, -0.0015070, 0.0016432
3: 0.0003317, 0.0007770, 0.0003213, 0.0007673, -0.0001994, 0.0002174
4: 0.0108940, 0.0134083, 0.0109487, 0.0134675, -0.0012280, 0.0011263
5: 0.9985330, 0.9992315, 0.9985482, 0.9992480, -0.0003412, 0.0003129
6: 0.0065520, 0.0071860, 0.0065658, 0.0072010, -0.0003097, 0.0002840
7: 0.0010693, 0.0034356, 0.0011208, 0.0034913, -0.0011557, 0.0010599
8: -0.0118668, -0.0100251, -0.0119102, -0.0100652, -0.0008250, 0.0008995
9: -0.0031448, -0.0029859, -0.0031414, -0.0029822, -0.0000776, 0.0000712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002390
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002390
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0128724, -0.0112368, -0.0007621, 0.0007927
1: -0.0065720, -0.0061090, -0.0065679, -0.0061067, -0.0002149, 0.0002235
2: -0.0099299, -0.0065137, -0.0098994, -0.0064969, -0.0015853, 0.0016489
3: 0.0003132, 0.0007653, 0.0003173, 0.0007675, -0.0002098, 0.0002182
4: 0.0109598, 0.0135129, 0.0109473, 0.0134900, -0.0012323, 0.0011847
5: 0.9985512, 0.9992605, 0.9985477, 0.9992542, -0.0003424, 0.0003292
6: 0.0065686, 0.0072124, 0.0065654, 0.0072067, -0.0003108, 0.0002988
7: 0.0011312, 0.0035340, 0.0011195, 0.0035125, -0.0011597, 0.0011150
8: -0.0119434, -0.0100733, -0.0119266, -0.0100641, -0.0008678, 0.0009026
9: -0.0031407, -0.0029793, -0.0031415, -0.0029808, -0.0000779, 0.0000749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002487
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002498
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0128187, -0.0112024, -0.0008128, 0.0007246
1: -0.0065651, -0.0061058, -0.0065527, -0.0060970, -0.0002291, 0.0002043
2: -0.0098789, -0.0064902, -0.0097877, -0.0064255, -0.0016907, 0.0015074
3: 0.0003200, 0.0007684, 0.0003320, 0.0007770, -0.0002237, 0.0001995
4: 0.0109422, 0.0134748, 0.0108939, 0.0134066, -0.0011265, 0.0012635
5: 0.9985464, 0.9992499, 0.9985330, 0.9992310, -0.0003130, 0.0003510
6: 0.0065641, 0.0072028, 0.0065519, 0.0071856, -0.0002841, 0.0003186
7: 0.0011147, 0.0034981, 0.0010692, 0.0034340, -0.0010602, 0.0011891
8: -0.0119155, -0.0100604, -0.0118655, -0.0100250, -0.0009255, 0.0008251
9: -0.0031418, -0.0029817, -0.0031448, -0.0029860, -0.0000712, 0.0000798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002348
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002348
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128626, -0.0112335, -0.0128878, -0.0112447, -0.0007339, 0.0007624
1: -0.0065651, -0.0061058, -0.0065722, -0.0061090, -0.0002069, 0.0002150
2: -0.0098789, -0.0064902, -0.0099315, -0.0065135, -0.0015266, 0.0015860
3: 0.0003200, 0.0007684, 0.0003130, 0.0007653, -0.0002020, 0.0002099
4: 0.0109422, 0.0134748, 0.0109597, 0.0135141, -0.0011853, 0.0011409
5: 0.9985464, 0.9992499, 0.9985511, 0.9992608, -0.0003293, 0.0003170
6: 0.0065641, 0.0072028, 0.0065685, 0.0072127, -0.0002989, 0.0002877
7: 0.0011147, 0.0034981, 0.0011311, 0.0035351, -0.0011155, 0.0010737
8: -0.0119155, -0.0100604, -0.0119442, -0.0100732, -0.0008357, 0.0008682
9: -0.0031418, -0.0029817, -0.0031407, -0.0029792, -0.0000749, 0.0000721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002453
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002452
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0128701, -0.0112332, -0.0007707, 0.0007925
1: -0.0065720, -0.0061090, -0.0065672, -0.0061057, -0.0002173, 0.0002234
2: -0.0099299, -0.0065137, -0.0098946, -0.0064896, -0.0016032, 0.0016486
3: 0.0003132, 0.0007653, 0.0003179, 0.0007685, -0.0002122, 0.0002182
4: 0.0109598, 0.0135129, 0.0109418, 0.0134865, -0.0012320, 0.0011981
5: 0.9985512, 0.9992605, 0.9985462, 0.9992533, -0.0003423, 0.0003329
6: 0.0065686, 0.0072124, 0.0065640, 0.0072058, -0.0003107, 0.0003021
7: 0.0011312, 0.0035340, 0.0011143, 0.0035092, -0.0011595, 0.0011275
8: -0.0119434, -0.0100733, -0.0119241, -0.0100601, -0.0008776, 0.0009024
9: -0.0031407, -0.0029793, -0.0031418, -0.0029810, -0.0000779, 0.0000757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001886, upper bound: 0.0002424
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001886, upper bound: 0.0002497
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0128198, -0.0112025, -0.0128800, -0.0112454, -0.0007011, 0.0008001
1: -0.0065530, -0.0060971, -0.0065700, -0.0061092, -0.0001977, 0.0002256
2: -0.0097900, -0.0064256, -0.0099151, -0.0065149, -0.0014583, 0.0016643
3: 0.0003317, 0.0007770, 0.0003152, 0.0007652, -0.0001930, 0.0002202
4: 0.0108940, 0.0134083, 0.0109607, 0.0135018, -0.0012438, 0.0010899
5: 0.9985330, 0.9992315, 0.9985515, 0.9992574, -0.0003456, 0.0003028
6: 0.0065520, 0.0071860, 0.0065688, 0.0072096, -0.0003137, 0.0002749
7: 0.0010693, 0.0034356, 0.0011321, 0.0035236, -0.0011706, 0.0010257
8: -0.0118668, -0.0100251, -0.0119353, -0.0100740, -0.0007983, 0.0009111
9: -0.0031448, -0.0029859, -0.0031406, -0.0029800, -0.0000786, 0.0000689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002391
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002391
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0128871, -0.0112448, -0.0128949, -0.0112445, -0.0007392, 0.0008007
1: -0.0065720, -0.0061090, -0.0065742, -0.0061089, -0.0002084, 0.0002257
2: -0.0099299, -0.0065137, -0.0099462, -0.0065130, -0.0015376, 0.0016656
3: 0.0003132, 0.0007653, 0.0003111, 0.0007654, -0.0002035, 0.0002204
4: 0.0109598, 0.0135129, 0.0109593, 0.0135251, -0.0012447, 0.0011491
5: 0.9985512, 0.9992605, 0.9985511, 0.9992639, -0.0003458, 0.0003193
6: 0.0065686, 0.0072124, 0.0065684, 0.0072155, -0.0003139, 0.0002898
7: 0.0011312, 0.0035340, 0.0011308, 0.0035454, -0.0011714, 0.0010814
8: -0.0119434, -0.0100733, -0.0119523, -0.0100729, -0.0008417, 0.0009117
9: -0.0031407, -0.0029793, -0.0031407, -0.0029786, -0.0000787, 0.0000726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002487
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002498
time: 0.66 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.20 seconds
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002191
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002305
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002162
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002305
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002108
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002298
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002076
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002282
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002302
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002324
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002284
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002306
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002326
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002328
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002262
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002346
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002339
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002345
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002327
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002326
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002264
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002344
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002339
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002347
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002107
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002300
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002191
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001949, upper bound: 0.0002317
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002076
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002286
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002162
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001969, upper bound: 0.0002297
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002094, upper bound: 0.0001781
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002094, upper bound: 0.0001814
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002136, upper bound: 0.0002108
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002136, upper bound: 0.0002294
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002168, upper bound: 0.0001781
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002168, upper bound: 0.0001814
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002202, upper bound: 0.0002076
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002202, upper bound: 0.0002274
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002161
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002162
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002300
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002292, upper bound: 0.0002318
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002155
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002156
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002282
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002334, upper bound: 0.0002298
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0001835
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0001840
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0002324
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001907, upper bound: 0.0002324
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001951, upper bound: 0.0002262
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001951, upper bound: 0.0002342
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002286, upper bound: 0.0002338
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002286, upper bound: 0.0002343
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0001846
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0001850
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0002324
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001925, upper bound: 0.0002323
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001968, upper bound: 0.0002264
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001968, upper bound: 0.0002342
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002338
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002303, upper bound: 0.0002343
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002246
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002445
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002343
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001808, upper bound: 0.0002471
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002224
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002429
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002311
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001835, upper bound: 0.0002451
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002246
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001978, upper bound: 0.0002423
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002224
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002040, upper bound: 0.0002404
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002337
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002338
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002445
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002141, upper bound: 0.0002471
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002330
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002333
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002426
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002185, upper bound: 0.0002452
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002340
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002340
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002453
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001755, upper bound: 0.0002453
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002418
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001814, upper bound: 0.0002497
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002133, upper bound: 0.0002388
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002133, upper bound: 0.0002388
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002487
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002137, upper bound: 0.0002498
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002343
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002343
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002453
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001778, upper bound: 0.0002454
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002421
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001837, upper bound: 0.0002497
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002153, upper bound: 0.0002389
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002153, upper bound: 0.0002388
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002487
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002156, upper bound: 0.0002497
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002251
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002446
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002347
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001863, upper bound: 0.0002471
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002230
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002430
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002316
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001890, upper bound: 0.0002452
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002022, upper bound: 0.0002251
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002022, upper bound: 0.0002422
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002087, upper bound: 0.0002231
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002087, upper bound: 0.0002403
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002170, upper bound: 0.0002339
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002170, upper bound: 0.0002341
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002171, upper bound: 0.0002444
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002171, upper bound: 0.0002471
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002334
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002336
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002426
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002213, upper bound: 0.0002452
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001828, upper bound: 0.0002346
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001828, upper bound: 0.0002346
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001827, upper bound: 0.0002454
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001827, upper bound: 0.0002453
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001870, upper bound: 0.0002421
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001870, upper bound: 0.0002496
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002390
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002390
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002487
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002165, upper bound: 0.0002498
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002348
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002348
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002453
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001845, upper bound: 0.0002452
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001886, upper bound: 0.0002424
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0001886, upper bound: 0.0002497
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002391
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002391
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002487
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.20
Output dim: 5, lower bound: -0.0002183, upper bound: 0.0002498

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127374, -0.0112297, -0.0007282, 0.0006920
1: -0.0065363, -0.0061090, -0.0065298, -0.0061047, -0.0002053, 0.0001951
2: -0.0096667, -0.0065136, -0.0096186, -0.0064823, -0.0015149, 0.0014394
3: 0.0003481, 0.0007653, 0.0003544, 0.0007695, -0.0002005, 0.0001905
4: 0.0109598, 0.0133161, 0.0109363, 0.0132802, -0.0010757, 0.0011321
5: 0.9985512, 0.9992059, 0.9985447, 0.9991959, -0.0002989, 0.0003145
6: 0.0065686, 0.0071628, 0.0065626, 0.0071537, -0.0002713, 0.0002855
7: 0.0011312, 0.0033488, 0.0011091, 0.0033150, -0.0010124, 0.0010655
8: -0.0117993, -0.0100733, -0.0117729, -0.0100561, -0.0008293, 0.0007879
9: -0.0031407, -0.0029918, -0.0031421, -0.0029940, -0.0000680, 0.0000715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001697, upper bound: 0.0002265
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001802, upper bound: 0.0002310
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127610, -0.0112380, -0.0007290, 0.0007245
1: -0.0065363, -0.0061090, -0.0065364, -0.0061071, -0.0002055, 0.0002043
2: -0.0096667, -0.0065136, -0.0096675, -0.0064994, -0.0015165, 0.0015071
3: 0.0003481, 0.0007653, 0.0003480, 0.0007672, -0.0002007, 0.0001994
4: 0.0109598, 0.0133161, 0.0109491, 0.0133168, -0.0011263, 0.0011334
5: 0.9985512, 0.9992059, 0.9985483, 0.9992061, -0.0003129, 0.0003149
6: 0.0065686, 0.0071628, 0.0065659, 0.0071630, -0.0002840, 0.0002858
7: 0.0011312, 0.0033488, 0.0011212, 0.0033494, -0.0010600, 0.0010666
8: -0.0117993, -0.0100733, -0.0117997, -0.0100655, -0.0008302, 0.0008250
9: -0.0031407, -0.0029918, -0.0031413, -0.0029917, -0.0000712, 0.0000716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001724, upper bound: 0.0002261
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001827, upper bound: 0.0002291
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0126867, -0.0112058, -0.0007727, 0.0006558
1: -0.0065363, -0.0061090, -0.0065155, -0.0060980, -0.0002178, 0.0001849
2: -0.0096667, -0.0065136, -0.0095132, -0.0064325, -0.0016073, 0.0013643
3: 0.0003481, 0.0007653, 0.0003684, 0.0007761, -0.0002127, 0.0001805
4: 0.0109598, 0.0133161, 0.0108991, 0.0132014, -0.0010196, 0.0012012
5: 0.9985512, 0.9992059, 0.9985344, 0.9991740, -0.0002833, 0.0003337
6: 0.0065686, 0.0071628, 0.0065533, 0.0071339, -0.0002571, 0.0003029
7: 0.0011312, 0.0033488, 0.0010741, 0.0032409, -0.0009595, 0.0011305
8: -0.0117993, -0.0100733, -0.0117152, -0.0100289, -0.0008799, 0.0007468
9: -0.0031407, -0.0029918, -0.0031445, -0.0029990, -0.0000644, 0.0000759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002035, upper bound: 0.0002169
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002129, upper bound: 0.0002289
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127605, -0.0112448, -0.0006966, 0.0006966
1: -0.0065363, -0.0061090, -0.0065363, -0.0061090, -0.0001964, 0.0001964
2: -0.0096667, -0.0065136, -0.0096667, -0.0065136, -0.0014490, 0.0014490
3: 0.0003481, 0.0007653, 0.0003481, 0.0007653, -0.0001918, 0.0001918
4: 0.0109598, 0.0133161, 0.0109598, 0.0133161, -0.0010829, 0.0010829
5: 0.9985512, 0.9992059, 0.9985512, 0.9992059, -0.0003009, 0.0003009
6: 0.0065686, 0.0071628, 0.0065686, 0.0071628, -0.0002731, 0.0002731
7: 0.0011312, 0.0033488, 0.0011312, 0.0033488, -0.0010191, 0.0010191
8: -0.0117993, -0.0100733, -0.0117993, -0.0100733, -0.0007932, 0.0007932
9: -0.0031407, -0.0029918, -0.0031407, -0.0029918, -0.0000684, 0.0000684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002035, upper bound: 0.0002278
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002129, upper bound: 0.0002309
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127605, -0.0112448, -0.0127808, -0.0112556, -0.0006967, 0.0007284
1: -0.0065363, -0.0061090, -0.0065420, -0.0061120, -0.0001964, 0.0002054
2: -0.0096667, -0.0065136, -0.0097088, -0.0065360, -0.0014493, 0.0015152
3: 0.0003481, 0.0007653, 0.0003425, 0.0007624, -0.0001918, 0.0002005
4: 0.0109598, 0.0133161, 0.0109765, 0.0133476, -0.0011324, 0.0010831
5: 0.9985512, 0.9992059, 0.9985558, 0.9992145, -0.0003146, 0.0003009
6: 0.0065686, 0.0071628, 0.0065728, 0.0071707, -0.0002856, 0.0002731
7: 0.0011312, 0.0033488, 0.0011469, 0.0033784, -0.0010657, 0.0010193
8: -0.0117993, -0.0100733, -0.0118223, -0.0100855, -0.0007933, 0.0008294
9: -0.0031407, -0.0029918, -0.0031396, -0.0029898, -0.0000716, 0.0000684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002058, upper bound: 0.0002271
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002173, upper bound: 0.0002292
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127610, -0.0112380, -0.0127374, -0.0112297, -0.0007223, 0.0006905
1: -0.0065364, -0.0061071, -0.0065298, -0.0061047, -0.0002036, 0.0001947
2: -0.0096675, -0.0064994, -0.0096186, -0.0064823, -0.0015025, 0.0014364
3: 0.0003480, 0.0007672, 0.0003544, 0.0007695, -0.0001988, 0.0001901
4: 0.0109491, 0.0133168, 0.0109363, 0.0132802, -0.0010735, 0.0011228
5: 0.9985483, 0.9992061, 0.9985447, 0.9991959, -0.0002982, 0.0003120
6: 0.0065659, 0.0071630, 0.0065626, 0.0071537, -0.0002707, 0.0002832
7: 0.0011212, 0.0033494, 0.0011091, 0.0033150, -0.0010102, 0.0010567
8: -0.0117997, -0.0100655, -0.0117729, -0.0100561, -0.0008224, 0.0007863
9: -0.0031413, -0.0029917, -0.0031421, -0.0029940, -0.0000678, 0.0000710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002239, upper bound: 0.0002263
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002241, upper bound: 0.0002312
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127610, -0.0112380, -0.0127605, -0.0112448, -0.0007245, 0.0007290
1: -0.0065364, -0.0061071, -0.0065363, -0.0061090, -0.0002043, 0.0002055
2: -0.0096675, -0.0064994, -0.0096667, -0.0065136, -0.0015071, 0.0015165
3: 0.0003480, 0.0007672, 0.0003481, 0.0007653, -0.0001994, 0.0002007
4: 0.0109491, 0.0133168, 0.0109598, 0.0133161, -0.0011334, 0.0011263
5: 0.9985483, 0.9992061, 0.9985512, 0.9992059, -0.0003149, 0.0003129
6: 0.0065659, 0.0071630, 0.0065686, 0.0071628, -0.0002858, 0.0002840
7: 0.0011212, 0.0033494, 0.0011312, 0.0033488, -0.0010666, 0.0010600
8: -0.0117997, -0.0100655, -0.0117993, -0.0100733, -0.0008250, 0.0008302
9: -0.0031413, -0.0029917, -0.0031407, -0.0029918, -0.0000716, 0.0000712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0002239, upper bound: 0.0002263
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0002241, upper bound: 0.0002313
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0127374, -0.0112297, -0.0007587, 0.0006908
1: -0.0065420, -0.0061120, -0.0065298, -0.0061047, -0.0002139, 0.0001948
2: -0.0097088, -0.0065360, -0.0096186, -0.0064823, -0.0015783, 0.0014370
3: 0.0003425, 0.0007624, 0.0003544, 0.0007695, -0.0002089, 0.0001902
4: 0.0109765, 0.0133476, 0.0109363, 0.0132802, -0.0010739, 0.0011795
5: 0.9985558, 0.9992145, 0.9985447, 0.9991959, -0.0002984, 0.0003277
6: 0.0065728, 0.0071707, 0.0065626, 0.0071537, -0.0002708, 0.0002975
7: 0.0011469, 0.0033784, 0.0011091, 0.0033150, -0.0010107, 0.0011101
8: -0.0118223, -0.0100855, -0.0117729, -0.0100561, -0.0008640, 0.0007866
9: -0.0031396, -0.0029898, -0.0031421, -0.0029940, -0.0000679, 0.0000745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001691, upper bound: 0.0002287
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001802, upper bound: 0.0002331
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0127808, -0.0112556, -0.0126867, -0.0112058, -0.0008073, 0.0006574
1: -0.0065420, -0.0061120, -0.0065155, -0.0060980, -0.0002276, 0.0001853
2: -0.0097088, -0.0065360, -0.0095132, -0.0064325, -0.0016793, 0.0013675
3: 0.0003425, 0.0007624, 0.0003684, 0.0007761, -0.0002222, 0.0001810
4: 0.0109765, 0.0133476, 0.0108991, 0.0132014, -0.0010220, 0.0012550
5: 0.9985558, 0.9992145, 0.9985344, 0.9991740, -0.0002839, 0.0003487
6: 0.0065728, 0.0071707, 0.0065533, 0.0071339, -0.0002577, 0.0003165
7: 0.0011469, 0.0033784, 0.0010741, 0.0032409, -0.0009618, 0.0011811
8: -0.0118223, -0.0100855, -0.0117152, -0.0100289, -0.0009192, 0.0007486
9: -0.0031396, -0.0029898, -0.0031445, -0.0029990, -0.0000646, 0.0000793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.05 + 597.50 = 600.55 seconds
