## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00085666


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9936841, 0.9965446, 0.9936841, 0.9965446, -0.0021934, 0.0021934)
1: (-0.0028377, -0.0021250, -0.0028377, -0.0021250, -0.0005465, 0.0005465)
2: (0.0012072, 0.0049845, 0.0012072, 0.0049845, -0.0028963, 0.0028963)
3: (-0.0035418, -0.0018226, -0.0035418, -0.0018226, -0.0013183, 0.0013183)
4: (0.0007615, 0.0014926, 0.0007615, 0.0014926, -0.0005606, 0.0005606)
5: (0.0004777, 0.0052286, 0.0004777, 0.0052286, -0.0036428, 0.0036428)
6: (0.0002138, 0.0014196, 0.0002138, 0.0014196, -0.0009246, 0.0009246)
7: (-0.0025846, 0.0005352, -0.0025846, 0.0005352, -0.0023922, 0.0023922)
8: (-0.0009233, 0.0007173, -0.0009233, 0.0007173, -0.0012580, 0.0012580)
9: (-0.0026956, -0.0007932, -0.0026956, -0.0007932, -0.0014588, 0.0014588)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.81 + 1.48 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0012341

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011485, upper bound: 0.0010894
time: 0.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011492, upper bound: 0.0011493
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -0.0011485, upper bound: 0.0010894
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -0.0011492, upper bound: 0.0011493

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9937093, 0.9962022, 0.9936907, 0.9964324, -0.0019150, 0.0017866
1: -0.0028314, -0.0022103, -0.0028361, -0.0021529, -0.0004772, 0.0004452
2: 0.0016593, 0.0049510, 0.0013552, 0.0049757, -0.0023592, 0.0025287
3: -0.0035266, -0.0020283, -0.0035378, -0.0018899, -0.0011509, 0.0010738
4: 0.0008490, 0.0014862, 0.0007902, 0.0014909, -0.0004566, 0.0004894
5: 0.0010464, 0.0051866, 0.0006639, 0.0052176, -0.0029673, 0.0031804
6: 0.0002244, 0.0012753, 0.0002166, 0.0013723, -0.0008072, 0.0007531
7: -0.0025570, 0.0001619, -0.0025773, 0.0004130, -0.0020885, 0.0019486
8: -0.0009088, 0.0005210, -0.0009195, 0.0006530, -0.0010983, 0.0010247
9: -0.0024679, -0.0008100, -0.0026211, -0.0007976, -0.0011882, 0.0012736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0010894
time: 0.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0010894
time: 0.76 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9936927, 0.9964258, 0.9936858, 0.9965202, -0.0021505, 0.0017904
1: -0.0028355, -0.0021545, -0.0028373, -0.0021310, -0.0005358, 0.0004461
2: 0.0013640, 0.0049729, 0.0012394, 0.0049822, -0.0023642, 0.0028397
3: -0.0035366, -0.0018939, -0.0035408, -0.0018372, -0.0012925, 0.0010761
4: 0.0007919, 0.0014904, 0.0007678, 0.0014922, -0.0004576, 0.0005496
5: 0.0006750, 0.0052141, 0.0005182, 0.0052258, -0.0029736, 0.0035716
6: 0.0002174, 0.0013695, 0.0002145, 0.0014093, -0.0009065, 0.0007547
7: -0.0025750, 0.0004057, -0.0025827, 0.0005087, -0.0023454, 0.0019527
8: -0.0009183, 0.0006492, -0.0009224, 0.0007033, -0.0012334, 0.0010269
9: -0.0026167, -0.0007990, -0.0026794, -0.0007943, -0.0011908, 0.0014302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0011486
time: 0.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0011492
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0010894
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0010894
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0011486
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 0, lower bound: -0.0010894, upper bound: 0.0011492

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937093, 0.9962022, 0.9937093, 0.9962022, -0.0016613, 0.0016613
1: -0.0028314, -0.0022103, -0.0028314, -0.0022103, -0.0004139, 0.0004139
2: 0.0016593, 0.0049510, 0.0016593, 0.0049510, -0.0021937, 0.0021937
3: -0.0035266, -0.0020283, -0.0035266, -0.0020283, -0.0009985, 0.0009985
4: 0.0008490, 0.0014862, 0.0008490, 0.0014862, -0.0004246, 0.0004246
5: 0.0010464, 0.0051866, 0.0010464, 0.0051866, -0.0027591, 0.0027591
6: 0.0002244, 0.0012753, 0.0002244, 0.0012753, -0.0007003, 0.0007003
7: -0.0025570, 0.0001619, -0.0025570, 0.0001619, -0.0018119, 0.0018119
8: -0.0009088, 0.0005210, -0.0009088, 0.0005210, -0.0009528, 0.0009528
9: -0.0024679, -0.0008100, -0.0024679, -0.0008100, -0.0011049, 0.0011049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0009544
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0010582
time: 0.66 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937093, 0.9962022, 0.9936927, 0.9964258, -0.0020018, 0.0017646
1: -0.0028314, -0.0022103, -0.0028355, -0.0021545, -0.0004988, 0.0004397
2: 0.0016593, 0.0049510, 0.0013640, 0.0049729, -0.0023301, 0.0026434
3: -0.0035266, -0.0020283, -0.0035366, -0.0018939, -0.0012032, 0.0010606
4: 0.0008490, 0.0014862, 0.0007919, 0.0014904, -0.0004510, 0.0005116
5: 0.0010464, 0.0051866, 0.0006750, 0.0052141, -0.0029306, 0.0033247
6: 0.0002244, 0.0012753, 0.0002174, 0.0013695, -0.0008438, 0.0007438
7: -0.0025570, 0.0001619, -0.0025750, 0.0004057, -0.0021833, 0.0019245
8: -0.0009088, 0.0005210, -0.0009183, 0.0006492, -0.0011482, 0.0010121
9: -0.0024679, -0.0008100, -0.0026167, -0.0007990, -0.0011736, 0.0013314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0010565
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0010582
time: 0.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9936927, 0.9964258, 0.9937093, 0.9962022, -0.0017646, 0.0020018
1: -0.0028355, -0.0021545, -0.0028314, -0.0022103, -0.0004397, 0.0004988
2: 0.0013640, 0.0049729, 0.0016593, 0.0049510, -0.0026434, 0.0023301
3: -0.0035366, -0.0018939, -0.0035266, -0.0020283, -0.0010606, 0.0012032
4: 0.0007919, 0.0014904, 0.0008490, 0.0014862, -0.0005116, 0.0004510
5: 0.0006750, 0.0052141, 0.0010464, 0.0051866, -0.0033247, 0.0029306
6: 0.0002174, 0.0013695, 0.0002244, 0.0012753, -0.0007438, 0.0008438
7: -0.0025750, 0.0004057, -0.0025570, 0.0001619, -0.0019245, 0.0021833
8: -0.0009183, 0.0006492, -0.0009088, 0.0005210, -0.0010121, 0.0011482
9: -0.0026167, -0.0007990, -0.0024679, -0.0008100, -0.0013314, 0.0011736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010192
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0011133
time: 0.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9936927, 0.9964258, 0.9936927, 0.9964258, -0.0017734, 0.0017734
1: -0.0028355, -0.0021545, -0.0028355, -0.0021545, -0.0004419, 0.0004419
2: 0.0013640, 0.0049729, 0.0013640, 0.0049729, -0.0023418, 0.0023418
3: -0.0035366, -0.0018939, -0.0035366, -0.0018939, -0.0010659, 0.0010659
4: 0.0007919, 0.0014904, 0.0007919, 0.0014904, -0.0004532, 0.0004532
5: 0.0006750, 0.0052141, 0.0006750, 0.0052141, -0.0029453, 0.0029453
6: 0.0002174, 0.0013695, 0.0002174, 0.0013695, -0.0007476, 0.0007476
7: -0.0025750, 0.0004057, -0.0025750, 0.0004057, -0.0019342, 0.0019342
8: -0.0009183, 0.0006492, -0.0009183, 0.0006492, -0.0010172, 0.0010172
9: -0.0026167, -0.0007990, -0.0026167, -0.0007990, -0.0011794, 0.0011794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010196
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0011139
time: 0.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0009544
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0010582
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0010565
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0010582
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010192
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0011133
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010196
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0010582, upper bound: 0.0011139

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9937546, 0.9959773, 0.9937112, 0.9961264, -0.0015469, 0.0014435
1: -0.0028201, -0.0022663, -0.0028310, -0.0022292, -0.0003854, 0.0003597
2: 0.0019562, 0.0048912, 0.0017593, 0.0049486, -0.0019062, 0.0020426
3: -0.0034994, -0.0021635, -0.0035255, -0.0020739, -0.0009297, 0.0008676
4: 0.0009065, 0.0014746, 0.0008684, 0.0014857, -0.0003689, 0.0003953
5: 0.0014198, 0.0051113, 0.0011722, 0.0051835, -0.0023975, 0.0025691
6: 0.0002435, 0.0011805, 0.0002252, 0.0012433, -0.0006521, 0.0006085
7: -0.0025076, -0.0000834, -0.0025549, 0.0000792, -0.0016871, 0.0015744
8: -0.0008829, 0.0003920, -0.0009078, 0.0004775, -0.0008872, 0.0008280
9: -0.0023184, -0.0008401, -0.0024175, -0.0008113, -0.0009601, 0.0010288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0009557
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0009557
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937117, 0.9961612, 0.9937097, 0.9961946, -0.0016536, 0.0014681
1: -0.0028309, -0.0022205, -0.0028313, -0.0022122, -0.0004120, 0.0003658
2: 0.0017134, 0.0049480, 0.0016694, 0.0049505, -0.0019386, 0.0021836
3: -0.0035253, -0.0020530, -0.0035264, -0.0020330, -0.0009939, 0.0008824
4: 0.0008595, 0.0014856, 0.0008510, 0.0014860, -0.0003752, 0.0004226
5: 0.0011144, 0.0051828, 0.0010591, 0.0051859, -0.0024383, 0.0027464
6: 0.0002254, 0.0012580, 0.0002246, 0.0012720, -0.0006971, 0.0006189
7: -0.0025545, 0.0001172, -0.0025565, 0.0001535, -0.0018035, 0.0016012
8: -0.0009075, 0.0004975, -0.0009086, 0.0005166, -0.0009484, 0.0008421
9: -0.0024407, -0.0008115, -0.0024628, -0.0008103, -0.0009764, 0.0010998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0010565
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0010582
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9937112, 0.9961264, 0.9937233, 0.9962045, -0.0017989, 0.0016737
1: -0.0028310, -0.0022292, -0.0028280, -0.0022097, -0.0004482, 0.0004170
2: 0.0017593, 0.0049486, 0.0016562, 0.0049327, -0.0022101, 0.0023755
3: -0.0035255, -0.0020739, -0.0035183, -0.0020269, -0.0010812, 0.0010060
4: 0.0008684, 0.0014857, 0.0008484, 0.0014826, -0.0004278, 0.0004598
5: 0.0011722, 0.0051835, 0.0010425, 0.0051634, -0.0027798, 0.0029877
6: 0.0002252, 0.0012433, 0.0002303, 0.0012762, -0.0007583, 0.0007055
7: -0.0025549, 0.0000792, -0.0025418, 0.0001644, -0.0019620, 0.0018254
8: -0.0009078, 0.0004775, -0.0009008, 0.0005223, -0.0010318, 0.0009600
9: -0.0024175, -0.0008113, -0.0024695, -0.0008193, -0.0011131, 0.0011964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010192, upper bound: 0.0009544
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010192, upper bound: 0.0010565
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9937097, 0.9961946, 0.9936956, 0.9963784, -0.0018242, 0.0017557
1: -0.0028313, -0.0022122, -0.0028348, -0.0021664, -0.0004545, 0.0004375
2: 0.0016694, 0.0049505, 0.0014267, 0.0049692, -0.0023183, 0.0024088
3: -0.0035264, -0.0020330, -0.0035349, -0.0019225, -0.0010964, 0.0010552
4: 0.0008510, 0.0014860, 0.0008040, 0.0014897, -0.0004487, 0.0004662
5: 0.0010591, 0.0051859, 0.0007538, 0.0052094, -0.0029159, 0.0030297
6: 0.0002246, 0.0012720, 0.0002186, 0.0013495, -0.0007690, 0.0007401
7: -0.0025565, 0.0001535, -0.0025720, 0.0003540, -0.0019895, 0.0019148
8: -0.0009086, 0.0005166, -0.0009167, 0.0006220, -0.0010463, 0.0010070
9: -0.0024628, -0.0008103, -0.0025851, -0.0008009, -0.0011676, 0.0012132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011110, upper bound: 0.0009544
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011110, upper bound: 0.0010582
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9937233, 0.9962045, 0.9937112, 0.9961264, -0.0016737, 0.0017989
1: -0.0028280, -0.0022097, -0.0028310, -0.0022292, -0.0004170, 0.0004482
2: 0.0016562, 0.0049327, 0.0017593, 0.0049486, -0.0023755, 0.0022101
3: -0.0035183, -0.0020269, -0.0035255, -0.0020739, -0.0010060, 0.0010812
4: 0.0008484, 0.0014826, 0.0008684, 0.0014857, -0.0004598, 0.0004278
5: 0.0010425, 0.0051634, 0.0011722, 0.0051835, -0.0029877, 0.0027798
6: 0.0002303, 0.0012762, 0.0002252, 0.0012433, -0.0007055, 0.0007583
7: -0.0025418, 0.0001644, -0.0025549, 0.0000792, -0.0018254, 0.0019620
8: -0.0009008, 0.0005223, -0.0009078, 0.0004775, -0.0009600, 0.0010318
9: -0.0024695, -0.0008193, -0.0024175, -0.0008113, -0.0011964, 0.0011131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0010192
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0010192
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9936956, 0.9963784, 0.9937097, 0.9961946, -0.0017557, 0.0018242
1: -0.0028348, -0.0021664, -0.0028313, -0.0022122, -0.0004375, 0.0004545
2: 0.0014267, 0.0049692, 0.0016694, 0.0049505, -0.0024088, 0.0023183
3: -0.0035349, -0.0019225, -0.0035264, -0.0020330, -0.0010552, 0.0010964
4: 0.0008040, 0.0014897, 0.0008510, 0.0014860, -0.0004662, 0.0004487
5: 0.0007538, 0.0052094, 0.0010591, 0.0051859, -0.0030297, 0.0029159
6: 0.0002186, 0.0013495, 0.0002246, 0.0012720, -0.0007401, 0.0007690
7: -0.0025720, 0.0003540, -0.0025565, 0.0001535, -0.0019148, 0.0019895
8: -0.0009167, 0.0006220, -0.0009086, 0.0005166, -0.0010070, 0.0010463
9: -0.0025851, -0.0008009, -0.0024628, -0.0008103, -0.0012132, 0.0011676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0011110
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0011133
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9937233, 0.9962045, 0.9936950, 0.9963524, -0.0016673, 0.0015566
1: -0.0028280, -0.0022097, -0.0028350, -0.0021728, -0.0004155, 0.0003879
2: 0.0016562, 0.0049327, 0.0014609, 0.0049700, -0.0020555, 0.0022017
3: -0.0035183, -0.0020269, -0.0035352, -0.0019381, -0.0010021, 0.0009356
4: 0.0008484, 0.0014826, 0.0008106, 0.0014898, -0.0003978, 0.0004261
5: 0.0010425, 0.0051634, 0.0007969, 0.0052104, -0.0025853, 0.0027691
6: 0.0002303, 0.0012762, 0.0002184, 0.0013386, -0.0007028, 0.0006562
7: -0.0025418, 0.0001644, -0.0025726, 0.0003257, -0.0018185, 0.0016977
8: -0.0009008, 0.0005223, -0.0009171, 0.0006071, -0.0009563, 0.0008928
9: -0.0024695, -0.0008193, -0.0025678, -0.0008005, -0.0010353, 0.0011089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0010196
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0010196
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9936956, 0.9963784, 0.9936933, 0.9964172, -0.0017655, 0.0015837
1: -0.0028348, -0.0021664, -0.0028354, -0.0021567, -0.0004399, 0.0003946
2: 0.0014267, 0.0049692, 0.0013753, 0.0049722, -0.0020913, 0.0023314
3: -0.0035349, -0.0019225, -0.0035363, -0.0018991, -0.0010611, 0.0009519
4: 0.0008040, 0.0014897, 0.0007941, 0.0014903, -0.0004048, 0.0004512
5: 0.0007538, 0.0052094, 0.0006893, 0.0052132, -0.0026303, 0.0029322
6: 0.0002186, 0.0013495, 0.0002177, 0.0013659, -0.0007442, 0.0006676
7: -0.0025720, 0.0003540, -0.0025745, 0.0003964, -0.0019256, 0.0017273
8: -0.0009167, 0.0006220, -0.0009180, 0.0006443, -0.0010126, 0.0009084
9: -0.0025851, -0.0008009, -0.0026109, -0.0007993, -0.0010533, 0.0011742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0011114
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0011139
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.60 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0009557
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0009557
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0010565
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009557, upper bound: 0.0010582
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0010192, upper bound: 0.0009544
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0010192, upper bound: 0.0010565
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0011110, upper bound: 0.0009544
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0011110, upper bound: 0.0010582
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0010192
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0010192
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0011110
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009544, upper bound: 0.0011133
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0010196
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0010196
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0011114
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.60
Output dim: 0, lower bound: -0.0009558, upper bound: 0.0011139

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937546, 0.9959773, 0.9937546, 0.9959773, -0.0014093, 0.0014093
1: -0.0028201, -0.0022663, -0.0028201, -0.0022663, -0.0003512, 0.0003512
2: 0.0019562, 0.0048912, 0.0019562, 0.0048912, -0.0018609, 0.0018609
3: -0.0034994, -0.0021635, -0.0034994, -0.0021635, -0.0008470, 0.0008470
4: 0.0009065, 0.0014746, 0.0009065, 0.0014746, -0.0003602, 0.0003602
5: 0.0014198, 0.0051113, 0.0014198, 0.0051113, -0.0023406, 0.0023406
6: 0.0002435, 0.0011805, 0.0002435, 0.0011805, -0.0005941, 0.0005941
7: -0.0025076, -0.0000834, -0.0025076, -0.0000834, -0.0015370, 0.0015370
8: -0.0008829, 0.0003920, -0.0008829, 0.0003920, -0.0008083, 0.0008083
9: -0.0023184, -0.0008401, -0.0023184, -0.0008401, -0.0009373, 0.0009373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009554, upper bound: 0.0009224
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009240
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937546, 0.9959773, 0.9937117, 0.9961612, -0.0015956, 0.0014432
1: -0.0028201, -0.0022663, -0.0028309, -0.0022205, -0.0003976, 0.0003596
2: 0.0019562, 0.0048912, 0.0017134, 0.0049480, -0.0019057, 0.0021069
3: -0.0034994, -0.0021635, -0.0035253, -0.0020530, -0.0009590, 0.0008674
4: 0.0009065, 0.0014746, 0.0008595, 0.0014856, -0.0003689, 0.0004078
5: 0.0014198, 0.0051113, 0.0011144, 0.0051828, -0.0023969, 0.0026500
6: 0.0002435, 0.0011805, 0.0002254, 0.0012580, -0.0006726, 0.0006084
7: -0.0025076, -0.0000834, -0.0025545, 0.0001172, -0.0017402, 0.0015740
8: -0.0008829, 0.0003920, -0.0009075, 0.0004975, -0.0009151, 0.0008278
9: -0.0023184, -0.0008401, -0.0024407, -0.0008115, -0.0009598, 0.0010612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009554, upper bound: 0.0009224
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009240
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937117, 0.9961612, 0.9937546, 0.9959773, -0.0014432, 0.0015956
1: -0.0028309, -0.0022205, -0.0028201, -0.0022663, -0.0003596, 0.0003976
2: 0.0017134, 0.0049480, 0.0019562, 0.0048912, -0.0021069, 0.0019057
3: -0.0035253, -0.0020530, -0.0034994, -0.0021635, -0.0008674, 0.0009590
4: 0.0008595, 0.0014856, 0.0009065, 0.0014746, -0.0004078, 0.0003689
5: 0.0011144, 0.0051828, 0.0014198, 0.0051113, -0.0026500, 0.0023969
6: 0.0002254, 0.0012580, 0.0002435, 0.0011805, -0.0006084, 0.0006726
7: -0.0025545, 0.0001172, -0.0025076, -0.0000834, -0.0015740, 0.0017402
8: -0.0009075, 0.0004975, -0.0008829, 0.0003920, -0.0008278, 0.0009151
9: -0.0024407, -0.0008115, -0.0023184, -0.0008401, -0.0010612, 0.0009598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009224, upper bound: 0.0010266
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0010254
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937117, 0.9961612, 0.9937117, 0.9961612, -0.0014668, 0.0014668
1: -0.0028309, -0.0022205, -0.0028309, -0.0022205, -0.0003655, 0.0003655
2: 0.0017134, 0.0049480, 0.0017134, 0.0049480, -0.0019369, 0.0019369
3: -0.0035253, -0.0020530, -0.0035253, -0.0020530, -0.0008816, 0.0008816
4: 0.0008595, 0.0014856, 0.0008595, 0.0014856, -0.0003749, 0.0003749
5: 0.0011144, 0.0051828, 0.0011144, 0.0051828, -0.0024361, 0.0024361
6: 0.0002254, 0.0012580, 0.0002254, 0.0012580, -0.0006183, 0.0006183
7: -0.0025545, 0.0001172, -0.0025545, 0.0001172, -0.0015997, 0.0015997
8: -0.0009075, 0.0004975, -0.0009075, 0.0004975, -0.0008413, 0.0008413
9: -0.0024407, -0.0008115, -0.0024407, -0.0008115, -0.0009755, 0.0009755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009244, upper bound: 0.0010260
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0010274
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9937546, 0.9959773, 0.9937233, 0.9962045, -0.0017647, 0.0015361
1: -0.0028201, -0.0022663, -0.0028280, -0.0022097, -0.0004397, 0.0003828
2: 0.0019562, 0.0048912, 0.0016562, 0.0049327, -0.0020284, 0.0023302
3: -0.0034994, -0.0021635, -0.0035183, -0.0020269, -0.0010606, 0.0009233
4: 0.0009065, 0.0014746, 0.0008484, 0.0014826, -0.0003926, 0.0004510
5: 0.0014198, 0.0051113, 0.0010425, 0.0051634, -0.0025513, 0.0029308
6: 0.0002435, 0.0011805, 0.0002303, 0.0012762, -0.0007439, 0.0006475
7: -0.0025076, -0.0000834, -0.0025418, 0.0001644, -0.0019246, 0.0016754
8: -0.0008829, 0.0003920, -0.0009008, 0.0005223, -0.0010121, 0.0008811
9: -0.0023184, -0.0008401, -0.0024695, -0.0008193, -0.0010216, 0.0011736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0009551
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0009559
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937117, 0.9961612, 0.9937233, 0.9962045, -0.0017986, 0.0017224
1: -0.0028309, -0.0022205, -0.0028280, -0.0022097, -0.0004482, 0.0004292
2: 0.0017134, 0.0049480, 0.0016562, 0.0049327, -0.0022744, 0.0023751
3: -0.0035253, -0.0020530, -0.0035183, -0.0020269, -0.0010810, 0.0010352
4: 0.0008595, 0.0014856, 0.0008484, 0.0014826, -0.0004402, 0.0004597
5: 0.0011144, 0.0051828, 0.0010425, 0.0051634, -0.0028606, 0.0029872
6: 0.0002254, 0.0012580, 0.0002303, 0.0012762, -0.0007582, 0.0007261
7: -0.0025545, 0.0001172, -0.0025418, 0.0001644, -0.0019616, 0.0018785
8: -0.0009075, 0.0004975, -0.0009008, 0.0005223, -0.0010316, 0.0009879
9: -0.0024407, -0.0008115, -0.0024695, -0.0008193, -0.0011455, 0.0011962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010266
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0010254
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9937546, 0.9959773, 0.9936956, 0.9963784, -0.0019253, 0.0015453
1: -0.0028201, -0.0022663, -0.0028348, -0.0021664, -0.0004797, 0.0003850
2: 0.0019562, 0.0048912, 0.0014267, 0.0049692, -0.0020405, 0.0025423
3: -0.0034994, -0.0021635, -0.0035349, -0.0019225, -0.0011571, 0.0009287
4: 0.0009065, 0.0014746, 0.0008040, 0.0014897, -0.0003949, 0.0004921
5: 0.0014198, 0.0051113, 0.0007538, 0.0052094, -0.0025664, 0.0031975
6: 0.0002435, 0.0011805, 0.0002186, 0.0013495, -0.0008116, 0.0006514
7: -0.0025076, -0.0000834, -0.0025720, 0.0003540, -0.0020998, 0.0016853
8: -0.0008829, 0.0003920, -0.0009167, 0.0006220, -0.0011043, 0.0008863
9: -0.0023184, -0.0008401, -0.0025851, -0.0008009, -0.0010277, 0.0012804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0009235
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0009226
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937117, 0.9961612, 0.9936956, 0.9963784, -0.0018229, 0.0015984
1: -0.0028309, -0.0022205, -0.0028348, -0.0021664, -0.0004542, 0.0003983
2: 0.0017134, 0.0049480, 0.0014267, 0.0049692, -0.0021106, 0.0024071
3: -0.0035253, -0.0020530, -0.0035349, -0.0019225, -0.0010956, 0.0009607
4: 0.0008595, 0.0014856, 0.0008040, 0.0014897, -0.0004085, 0.0004659
5: 0.0011144, 0.0051828, 0.0007538, 0.0052094, -0.0026546, 0.0030275
6: 0.0002254, 0.0012580, 0.0002186, 0.0013495, -0.0007684, 0.0006738
7: -0.0025545, 0.0001172, -0.0025720, 0.0003540, -0.0019881, 0.0017432
8: -0.0009075, 0.0004975, -0.0009167, 0.0006220, -0.0010455, 0.0009167
9: -0.0024407, -0.0008115, -0.0025851, -0.0008009, -0.0010630, 0.0012123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010283
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0010274
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937233, 0.9962045, 0.9937546, 0.9959773, -0.0015361, 0.0017647
1: -0.0028280, -0.0022097, -0.0028201, -0.0022663, -0.0003828, 0.0004397
2: 0.0016562, 0.0049327, 0.0019562, 0.0048912, -0.0023302, 0.0020284
3: -0.0035183, -0.0020269, -0.0034994, -0.0021635, -0.0009233, 0.0010606
4: 0.0008484, 0.0014826, 0.0009065, 0.0014746, -0.0004510, 0.0003926
5: 0.0010425, 0.0051634, 0.0014198, 0.0051113, -0.0029308, 0.0025513
6: 0.0002303, 0.0012762, 0.0002435, 0.0011805, -0.0006475, 0.0007439
7: -0.0025418, 0.0001644, -0.0025076, -0.0000834, -0.0016754, 0.0019246
8: -0.0009008, 0.0005223, -0.0008829, 0.0003920, -0.0008811, 0.0010121
9: -0.0024695, -0.0008193, -0.0023184, -0.0008401, -0.0011736, 0.0010216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009551, upper bound: 0.0009870
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009559, upper bound: 0.0009882
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937233, 0.9962045, 0.9937117, 0.9961612, -0.0017224, 0.0017986
1: -0.0028280, -0.0022097, -0.0028309, -0.0022205, -0.0004292, 0.0004482
2: 0.0016562, 0.0049327, 0.0017134, 0.0049480, -0.0023751, 0.0022744
3: -0.0035183, -0.0020269, -0.0035253, -0.0020530, -0.0010352, 0.0010810
4: 0.0008484, 0.0014826, 0.0008595, 0.0014856, -0.0004597, 0.0004402
5: 0.0010425, 0.0051634, 0.0011144, 0.0051828, -0.0029872, 0.0028606
6: 0.0002303, 0.0012762, 0.0002254, 0.0012580, -0.0007261, 0.0007582
7: -0.0025418, 0.0001644, -0.0025545, 0.0001172, -0.0018785, 0.0019616
8: -0.0009008, 0.0005223, -0.0009075, 0.0004975, -0.0009879, 0.0010316
9: -0.0024695, -0.0008193, -0.0024407, -0.0008115, -0.0011962, 0.0011455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009551, upper bound: 0.0009870
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009559, upper bound: 0.0009882
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9936956, 0.9963784, 0.9937546, 0.9959773, -0.0015453, 0.0019253
1: -0.0028348, -0.0021664, -0.0028201, -0.0022663, -0.0003850, 0.0004797
2: 0.0014267, 0.0049692, 0.0019562, 0.0048912, -0.0025423, 0.0020405
3: -0.0035349, -0.0019225, -0.0034994, -0.0021635, -0.0009287, 0.0011571
4: 0.0008040, 0.0014897, 0.0009065, 0.0014746, -0.0004921, 0.0003949
5: 0.0007538, 0.0052094, 0.0014198, 0.0051113, -0.0031975, 0.0025664
6: 0.0002186, 0.0013495, 0.0002435, 0.0011805, -0.0006514, 0.0008116
7: -0.0025720, 0.0003540, -0.0025076, -0.0000834, -0.0016853, 0.0020998
8: -0.0009167, 0.0006220, -0.0008829, 0.0003920, -0.0008863, 0.0011043
9: -0.0025851, -0.0008009, -0.0023184, -0.0008401, -0.0012804, 0.0010277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009235, upper bound: 0.0010796
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009226, upper bound: 0.0010801
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9936956, 0.9963784, 0.9937117, 0.9961612, -0.0015984, 0.0018229
1: -0.0028348, -0.0021664, -0.0028309, -0.0022205, -0.0003983, 0.0004542
2: 0.0014267, 0.0049692, 0.0017134, 0.0049480, -0.0024071, 0.0021106
3: -0.0035349, -0.0019225, -0.0035253, -0.0020530, -0.0009607, 0.0010956
4: 0.0008040, 0.0014897, 0.0008595, 0.0014856, -0.0004659, 0.0004085
5: 0.0007538, 0.0052094, 0.0011144, 0.0051828, -0.0030275, 0.0026546
6: 0.0002186, 0.0013495, 0.0002254, 0.0012580, -0.0006738, 0.0007684
7: -0.0025720, 0.0003540, -0.0025545, 0.0001172, -0.0017432, 0.0019881
8: -0.0009167, 0.0006220, -0.0009075, 0.0004975, -0.0009167, 0.0010455
9: -0.0025851, -0.0008009, -0.0024407, -0.0008115, -0.0012123, 0.0010630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009235, upper bound: 0.0010819
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009226, upper bound: 0.0010827
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937233, 0.9962045, 0.9937233, 0.9962045, -0.0015303, 0.0015303
1: -0.0028280, -0.0022097, -0.0028280, -0.0022097, -0.0003813, 0.0003813
2: 0.0016562, 0.0049327, 0.0016562, 0.0049327, -0.0020208, 0.0020208
3: -0.0035183, -0.0020269, -0.0035183, -0.0020269, -0.0009198, 0.0009198
4: 0.0008484, 0.0014826, 0.0008484, 0.0014826, -0.0003911, 0.0003911
5: 0.0010425, 0.0051634, 0.0010425, 0.0051634, -0.0025416, 0.0025416
6: 0.0002303, 0.0012762, 0.0002303, 0.0012762, -0.0006451, 0.0006451
7: -0.0025418, 0.0001644, -0.0025418, 0.0001644, -0.0016690, 0.0016690
8: -0.0009008, 0.0005223, -0.0009008, 0.0005223, -0.0008777, 0.0008777
9: -0.0024695, -0.0008193, -0.0024695, -0.0008193, -0.0010178, 0.0010178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009587, upper bound: 0.0009875
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009582, upper bound: 0.0009884
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937233, 0.9962045, 0.9936956, 0.9963784, -0.0017157, 0.0015562
1: -0.0028280, -0.0022097, -0.0028348, -0.0021664, -0.0004275, 0.0003878
2: 0.0016562, 0.0049327, 0.0014267, 0.0049692, -0.0020550, 0.0022656
3: -0.0035183, -0.0020269, -0.0035349, -0.0019225, -0.0010312, 0.0009353
4: 0.0008484, 0.0014826, 0.0008040, 0.0014897, -0.0003977, 0.0004385
5: 0.0010425, 0.0051634, 0.0007538, 0.0052094, -0.0025846, 0.0028495
6: 0.0002303, 0.0012762, 0.0002186, 0.0013495, -0.0007232, 0.0006560
7: -0.0025418, 0.0001644, -0.0025720, 0.0003540, -0.0018712, 0.0016973
8: -0.0009008, 0.0005223, -0.0009167, 0.0006220, -0.0009841, 0.0008926
9: -0.0024695, -0.0008193, -0.0025851, -0.0008009, -0.0010350, 0.0011411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009587, upper bound: 0.0009875
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009582, upper bound: 0.0009883
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9936956, 0.9963784, 0.9937233, 0.9962045, -0.0015562, 0.0017157
1: -0.0028348, -0.0021664, -0.0028280, -0.0022097, -0.0003878, 0.0004275
2: 0.0014267, 0.0049692, 0.0016562, 0.0049327, -0.0022656, 0.0020550
3: -0.0035349, -0.0019225, -0.0035183, -0.0020269, -0.0009353, 0.0010312
4: 0.0008040, 0.0014897, 0.0008484, 0.0014826, -0.0004385, 0.0003977
5: 0.0007538, 0.0052094, 0.0010425, 0.0051634, -0.0028495, 0.0025846
6: 0.0002186, 0.0013495, 0.0002303, 0.0012762, -0.0006560, 0.0007232
7: -0.0025720, 0.0003540, -0.0025418, 0.0001644, -0.0016973, 0.0018712
8: -0.0009167, 0.0006220, -0.0009008, 0.0005223, -0.0008926, 0.0009841
9: -0.0025851, -0.0008009, -0.0024695, -0.0008193, -0.0011411, 0.0010350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009230, upper bound: 0.0010834
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009241, upper bound: 0.0010807
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9936956, 0.9963784, 0.9936956, 0.9963784, -0.0015824, 0.0015824
1: -0.0028348, -0.0021664, -0.0028348, -0.0021664, -0.0003943, 0.0003943
2: 0.0014267, 0.0049692, 0.0014267, 0.0049692, -0.0020896, 0.0020896
3: -0.0035349, -0.0019225, -0.0035349, -0.0019225, -0.0009511, 0.0009511
4: 0.0008040, 0.0014897, 0.0008040, 0.0014897, -0.0004044, 0.0004044
5: 0.0007538, 0.0052094, 0.0007538, 0.0052094, -0.0026282, 0.0026282
6: 0.0002186, 0.0013495, 0.0002186, 0.0013495, -0.0006671, 0.0006671
7: -0.0025720, 0.0003540, -0.0025720, 0.0003540, -0.0017259, 0.0017259
8: -0.0009167, 0.0006220, -0.0009167, 0.0006220, -0.0009076, 0.0009076
9: -0.0025851, -0.0008009, -0.0025851, -0.0008009, -0.0010524, 0.0010524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009256, upper bound: 0.0010826
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009241, upper bound: 0.0010834
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.33 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009554, upper bound: 0.0009224
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009240
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009554, upper bound: 0.0009224
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009240
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009224, upper bound: 0.0010266
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0010254
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009244, upper bound: 0.0010260
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009240, upper bound: 0.0010274
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0009551
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0009559
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010266
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0010254
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0009235
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0009226
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010283
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009882, upper bound: 0.0010274
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009551, upper bound: 0.0009870
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009559, upper bound: 0.0009882
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009551, upper bound: 0.0009870
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009559, upper bound: 0.0009882
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009235, upper bound: 0.0010796
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009226, upper bound: 0.0010801
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009235, upper bound: 0.0010819
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009226, upper bound: 0.0010827
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009587, upper bound: 0.0009875
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009582, upper bound: 0.0009884
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009587, upper bound: 0.0009875
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009582, upper bound: 0.0009883
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009230, upper bound: 0.0010834
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009241, upper bound: 0.0010807
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009256, upper bound: 0.0010826
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 0, lower bound: -0.0009241, upper bound: 0.0010834

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938462, 0.9959931, 0.9937886, 0.9959753, -0.0013013, 0.0013672
1: -0.0027973, -0.0022624, -0.0028116, -0.0022668, -0.0003243, 0.0003407
2: 0.0019353, 0.0047704, 0.0019588, 0.0048462, -0.0018054, 0.0017184
3: -0.0034444, -0.0021540, -0.0034789, -0.0021647, -0.0007821, 0.0008217
4: 0.0009025, 0.0014512, 0.0009070, 0.0014659, -0.0003494, 0.0003326
5: 0.0013936, 0.0049593, 0.0014231, 0.0050547, -0.0022707, 0.0021613
6: 0.0002821, 0.0011871, 0.0002579, 0.0011796, -0.0005486, 0.0005763
7: -0.0024077, -0.0000662, -0.0024704, -0.0000856, -0.0014193, 0.0014911
8: -0.0008303, 0.0004010, -0.0008633, 0.0003909, -0.0007464, 0.0007842
9: -0.0023289, -0.0009010, -0.0023171, -0.0008628, -0.0009093, 0.0008655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009195
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009304, upper bound: 0.0009288
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9937633, 0.9959769, -0.0013307, 0.0013992
1: -0.0028080, -0.0022669, -0.0028180, -0.0022664, -0.0003316, 0.0003486
2: 0.0019593, 0.0048270, 0.0019567, 0.0048799, -0.0018476, 0.0017572
3: -0.0034702, -0.0021649, -0.0034943, -0.0021638, -0.0007998, 0.0008410
4: 0.0009071, 0.0014621, 0.0009066, 0.0014724, -0.0003576, 0.0003401
5: 0.0014237, 0.0050306, 0.0014205, 0.0050971, -0.0023238, 0.0022100
6: 0.0002640, 0.0011795, 0.0002471, 0.0011803, -0.0005609, 0.0005898
7: -0.0024545, -0.0000860, -0.0024982, -0.0000839, -0.0014513, 0.0015260
8: -0.0008550, 0.0003907, -0.0008779, 0.0003917, -0.0007632, 0.0008025
9: -0.0023168, -0.0008725, -0.0023181, -0.0008458, -0.0009306, 0.0008850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009533, upper bound: 0.0009533
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009533, upper bound: 0.0009562
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938462, 0.9959931, 0.9937457, 0.9961591, -0.0014873, 0.0014013
1: -0.0027973, -0.0022624, -0.0028224, -0.0022210, -0.0003706, 0.0003492
2: 0.0019353, 0.0047704, 0.0017161, 0.0049030, -0.0018504, 0.0019639
3: -0.0034444, -0.0021540, -0.0035048, -0.0020542, -0.0008939, 0.0008422
4: 0.0009025, 0.0014512, 0.0008600, 0.0014769, -0.0003581, 0.0003801
5: 0.0013936, 0.0049593, 0.0011179, 0.0051262, -0.0023274, 0.0024701
6: 0.0002821, 0.0011871, 0.0002398, 0.0012571, -0.0006269, 0.0005907
7: -0.0024077, -0.0000662, -0.0025173, 0.0001149, -0.0016221, 0.0015283
8: -0.0008303, 0.0004010, -0.0008880, 0.0004963, -0.0008530, 0.0008037
9: -0.0023289, -0.0009010, -0.0024393, -0.0008342, -0.0009320, 0.0009891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009831, upper bound: 0.0008893
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0008982
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9937202, 0.9961608, -0.0015244, 0.0014334
1: -0.0028080, -0.0022669, -0.0028287, -0.0022206, -0.0003798, 0.0003572
2: 0.0019593, 0.0048270, 0.0017139, 0.0049367, -0.0018928, 0.0020129
3: -0.0034702, -0.0021649, -0.0035201, -0.0020532, -0.0009162, 0.0008615
4: 0.0009071, 0.0014621, 0.0008596, 0.0014834, -0.0003663, 0.0003896
5: 0.0014237, 0.0050306, 0.0011151, 0.0051685, -0.0023806, 0.0025318
6: 0.0002640, 0.0011795, 0.0002290, 0.0012578, -0.0006426, 0.0006042
7: -0.0024545, -0.0000860, -0.0025451, 0.0001167, -0.0016626, 0.0015633
8: -0.0008550, 0.0003907, -0.0009026, 0.0004972, -0.0008743, 0.0008221
9: -0.0023168, -0.0008725, -0.0024404, -0.0008173, -0.0009533, 0.0010138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009225
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009240
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9937457, 0.9961591, 0.9938462, 0.9959931, -0.0014013, 0.0014873
1: -0.0028224, -0.0022210, -0.0027973, -0.0022624, -0.0003492, 0.0003706
2: 0.0017161, 0.0049030, 0.0019353, 0.0047704, -0.0019639, 0.0018504
3: -0.0035048, -0.0020542, -0.0034444, -0.0021540, -0.0008422, 0.0008939
4: 0.0008600, 0.0014769, 0.0009025, 0.0014512, -0.0003801, 0.0003581
5: 0.0011179, 0.0051262, 0.0013936, 0.0049593, -0.0024701, 0.0023274
6: 0.0002398, 0.0012571, 0.0002821, 0.0011871, -0.0005907, 0.0006269
7: -0.0025173, 0.0001149, -0.0024077, -0.0000662, -0.0015283, 0.0016221
8: -0.0008880, 0.0004963, -0.0008303, 0.0004010, -0.0008037, 0.0008530
9: -0.0024393, -0.0008342, -0.0023289, -0.0009010, -0.0009891, 0.0009320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008893, upper bound: 0.0009832
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010018
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9937202, 0.9961608, 0.9938033, 0.9959750, -0.0014334, 0.0015244
1: -0.0028287, -0.0022206, -0.0028080, -0.0022669, -0.0003572, 0.0003798
2: 0.0017139, 0.0049367, 0.0019593, 0.0048270, -0.0020129, 0.0018928
3: -0.0035201, -0.0020532, -0.0034702, -0.0021649, -0.0008615, 0.0009162
4: 0.0008596, 0.0014834, 0.0009071, 0.0014621, -0.0003896, 0.0003663
5: 0.0011151, 0.0051685, 0.0014237, 0.0050306, -0.0025318, 0.0023806
6: 0.0002290, 0.0012578, 0.0002640, 0.0011795, -0.0006042, 0.0006426
7: -0.0025451, 0.0001167, -0.0024545, -0.0000860, -0.0015633, 0.0016626
8: -0.0009026, 0.0004972, -0.0008550, 0.0003907, -0.0008221, 0.0008743
9: -0.0024404, -0.0008173, -0.0023168, -0.0008725, -0.0010138, 0.0009533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009225, upper bound: 0.0010244
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009225, upper bound: 0.0010254
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938022, 0.9961815, 0.9937457, 0.9961591, -0.0013635, 0.0014278
1: -0.0028083, -0.0022154, -0.0028224, -0.0022210, -0.0003397, 0.0003558
2: 0.0016866, 0.0048283, 0.0017161, 0.0049030, -0.0018854, 0.0018004
3: -0.0034708, -0.0020408, -0.0035048, -0.0020542, -0.0008195, 0.0008581
4: 0.0008543, 0.0014624, 0.0008600, 0.0014769, -0.0003649, 0.0003485
5: 0.0010807, 0.0050322, 0.0011179, 0.0051262, -0.0023713, 0.0022645
6: 0.0002636, 0.0012665, 0.0002398, 0.0012571, -0.0005747, 0.0006019
7: -0.0024556, 0.0001393, -0.0025173, 0.0001149, -0.0014870, 0.0015572
8: -0.0008555, 0.0005091, -0.0008880, 0.0004963, -0.0007820, 0.0008189
9: -0.0024542, -0.0008718, -0.0024393, -0.0008342, -0.0009496, 0.0009068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008979, upper bound: 0.0009840
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0010021
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937605, 0.9961587, 0.9937202, 0.9961608, -0.0013913, 0.0014566
1: -0.0028187, -0.0022211, -0.0028287, -0.0022206, -0.0003467, 0.0003630
2: 0.0017166, 0.0048835, 0.0017139, 0.0049367, -0.0019235, 0.0018372
3: -0.0034959, -0.0020545, -0.0035201, -0.0020532, -0.0008362, 0.0008755
4: 0.0008601, 0.0014731, 0.0008596, 0.0014834, -0.0003723, 0.0003556
5: 0.0011185, 0.0051016, 0.0011151, 0.0051685, -0.0024192, 0.0023107
6: 0.0002460, 0.0012569, 0.0002290, 0.0012578, -0.0005865, 0.0006140
7: -0.0025011, 0.0001145, -0.0025451, 0.0001167, -0.0015174, 0.0015887
8: -0.0008795, 0.0004961, -0.0009026, 0.0004972, -0.0007980, 0.0008355
9: -0.0024391, -0.0008441, -0.0024404, -0.0008173, -0.0009688, 0.0009253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009289, upper bound: 0.0010261
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009289, upper bound: 0.0010273
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937886, 0.9959753, 0.9938119, 0.9962257, -0.0017306, 0.0014383
1: -0.0028116, -0.0022668, -0.0028059, -0.0022044, -0.0004312, 0.0003584
2: 0.0019588, 0.0048462, 0.0016282, 0.0048157, -0.0018992, 0.0022853
3: -0.0034789, -0.0021647, -0.0034650, -0.0020142, -0.0010402, 0.0008644
4: 0.0009070, 0.0014659, 0.0008430, 0.0014600, -0.0003676, 0.0004423
5: 0.0014231, 0.0050547, 0.0010073, 0.0050163, -0.0023887, 0.0028743
6: 0.0002579, 0.0011796, 0.0002676, 0.0012852, -0.0007295, 0.0006063
7: -0.0024704, -0.0000856, -0.0024452, 0.0001875, -0.0018875, 0.0015686
8: -0.0008633, 0.0003909, -0.0008500, 0.0005345, -0.0009926, 0.0008249
9: -0.0023171, -0.0008628, -0.0024836, -0.0008782, -0.0009565, 0.0011510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009740, upper bound: 0.0009206
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009909, upper bound: 0.0009300
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937633, 0.9959769, 0.9937711, 0.9962023, -0.0017548, 0.0014644
1: -0.0028180, -0.0022664, -0.0028160, -0.0022102, -0.0004373, 0.0003649
2: 0.0019567, 0.0048799, 0.0016591, 0.0048696, -0.0019338, 0.0023172
3: -0.0034943, -0.0021638, -0.0034895, -0.0020283, -0.0010547, 0.0008802
4: 0.0009066, 0.0014724, 0.0008490, 0.0014704, -0.0003743, 0.0004485
5: 0.0014205, 0.0050971, 0.0010461, 0.0050841, -0.0024322, 0.0029145
6: 0.0002471, 0.0011803, 0.0002504, 0.0012753, -0.0007397, 0.0006173
7: -0.0024982, -0.0000839, -0.0024897, 0.0001620, -0.0019139, 0.0015972
8: -0.0008779, 0.0003917, -0.0008734, 0.0005210, -0.0010065, 0.0008399
9: -0.0023181, -0.0008458, -0.0024680, -0.0008511, -0.0009739, 0.0011671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010165, upper bound: 0.0009532
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010165, upper bound: 0.0009559
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937457, 0.9961591, 0.9938119, 0.9962257, -0.0017648, 0.0016242
1: -0.0028224, -0.0022210, -0.0028059, -0.0022044, -0.0004397, 0.0004047
2: 0.0017161, 0.0049030, 0.0016282, 0.0048157, -0.0021447, 0.0023303
3: -0.0035048, -0.0020542, -0.0034650, -0.0020142, -0.0010607, 0.0009762
4: 0.0008600, 0.0014769, 0.0008430, 0.0014600, -0.0004151, 0.0004510
5: 0.0011179, 0.0051262, 0.0010073, 0.0050163, -0.0026975, 0.0029309
6: 0.0002398, 0.0012571, 0.0002676, 0.0012852, -0.0007439, 0.0006847
7: -0.0025173, 0.0001149, -0.0024452, 0.0001875, -0.0019247, 0.0017714
8: -0.0008880, 0.0004963, -0.0008500, 0.0005345, -0.0010122, 0.0009316
9: -0.0024393, -0.0008342, -0.0024836, -0.0008782, -0.0010802, 0.0011737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009444, upper bound: 0.0009937
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009613, upper bound: 0.0010018
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937202, 0.9961608, 0.9937711, 0.9962023, -0.0017890, 0.0016581
1: -0.0028287, -0.0022206, -0.0028160, -0.0022102, -0.0004458, 0.0004132
2: 0.0017139, 0.0049367, 0.0016591, 0.0048696, -0.0021895, 0.0023624
3: -0.0035201, -0.0020532, -0.0034895, -0.0020283, -0.0010753, 0.0009966
4: 0.0008596, 0.0014834, 0.0008490, 0.0014704, -0.0004238, 0.0004572
5: 0.0011151, 0.0051685, 0.0010461, 0.0050841, -0.0027539, 0.0029713
6: 0.0002290, 0.0012578, 0.0002504, 0.0012753, -0.0007541, 0.0006990
7: -0.0025451, 0.0001167, -0.0024897, 0.0001620, -0.0019512, 0.0018084
8: -0.0009026, 0.0004972, -0.0008734, 0.0005210, -0.0010261, 0.0009510
9: -0.0024404, -0.0008173, -0.0024680, -0.0008511, -0.0011028, 0.0011898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010244
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010254
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9937886, 0.9959753, 0.9937840, 0.9963966, -0.0018896, 0.0014450
1: -0.0028116, -0.0022668, -0.0028128, -0.0021618, -0.0004708, 0.0003601
2: 0.0019588, 0.0048462, 0.0014025, 0.0048524, -0.0019081, 0.0024952
3: -0.0034789, -0.0021647, -0.0034817, -0.0019115, -0.0011357, 0.0008685
4: 0.0009070, 0.0014659, 0.0007993, 0.0014671, -0.0003693, 0.0004829
5: 0.0014231, 0.0050547, 0.0007234, 0.0050625, -0.0023999, 0.0031384
6: 0.0002579, 0.0011796, 0.0002559, 0.0013572, -0.0007965, 0.0006091
7: -0.0024704, -0.0000856, -0.0024755, 0.0003740, -0.0020609, 0.0015760
8: -0.0008633, 0.0003909, -0.0008660, 0.0006325, -0.0010838, 0.0008288
9: -0.0023171, -0.0008628, -0.0025973, -0.0008597, -0.0009610, 0.0012567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010370, upper bound: 0.0008900
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008997
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937633, 0.9959769, 0.9937444, 0.9963761, -0.0019150, 0.0014707
1: -0.0028180, -0.0022664, -0.0028227, -0.0021670, -0.0004772, 0.0003665
2: 0.0019567, 0.0048799, 0.0014297, 0.0049048, -0.0019420, 0.0025287
3: -0.0034943, -0.0021638, -0.0035056, -0.0019239, -0.0011510, 0.0008839
4: 0.0009066, 0.0014724, 0.0008046, 0.0014772, -0.0003759, 0.0004894
5: 0.0014205, 0.0050971, 0.0007576, 0.0051284, -0.0024426, 0.0031805
6: 0.0002471, 0.0011803, 0.0002392, 0.0013485, -0.0008072, 0.0006200
7: -0.0024982, -0.0000839, -0.0025188, 0.0003514, -0.0020886, 0.0016040
8: -0.0008779, 0.0003917, -0.0008887, 0.0006207, -0.0010984, 0.0008435
9: -0.0023181, -0.0008458, -0.0025836, -0.0008333, -0.0009781, 0.0012736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010795, upper bound: 0.0009211
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010795, upper bound: 0.0009226
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937457, 0.9961591, 0.9937840, 0.9963966, -0.0017897, 0.0015023
1: -0.0028224, -0.0022210, -0.0028128, -0.0021618, -0.0004460, 0.0003743
2: 0.0017161, 0.0049030, 0.0014025, 0.0048524, -0.0019837, 0.0023633
3: -0.0035048, -0.0020542, -0.0034817, -0.0019115, -0.0010757, 0.0009029
4: 0.0008600, 0.0014769, 0.0007993, 0.0014671, -0.0003840, 0.0004574
5: 0.0011179, 0.0051262, 0.0007234, 0.0050625, -0.0024950, 0.0029724
6: 0.0002398, 0.0012571, 0.0002559, 0.0013572, -0.0007544, 0.0006333
7: -0.0025173, 0.0001149, -0.0024755, 0.0003740, -0.0019520, 0.0016385
8: -0.0008880, 0.0004963, -0.0008660, 0.0006325, -0.0010265, 0.0008616
9: -0.0024393, -0.0008342, -0.0025973, -0.0008597, -0.0009991, 0.0011903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009504, upper bound: 0.0009949
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009681, upper bound: 0.0010036
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937202, 0.9961608, 0.9937444, 0.9963761, -0.0018128, 0.0015289
1: -0.0028287, -0.0022206, -0.0028227, -0.0021670, -0.0004517, 0.0003810
2: 0.0017139, 0.0049367, 0.0014297, 0.0049048, -0.0020189, 0.0023938
3: -0.0035201, -0.0020532, -0.0035056, -0.0019239, -0.0010896, 0.0009189
4: 0.0008596, 0.0014834, 0.0008046, 0.0014772, -0.0003908, 0.0004633
5: 0.0011151, 0.0051685, 0.0007576, 0.0051284, -0.0025393, 0.0030108
6: 0.0002290, 0.0012578, 0.0002392, 0.0013485, -0.0007642, 0.0006445
7: -0.0025451, 0.0001167, -0.0025188, 0.0003514, -0.0019772, 0.0016675
8: -0.0009026, 0.0004972, -0.0008887, 0.0006207, -0.0010398, 0.0008769
9: -0.0024404, -0.0008173, -0.0025836, -0.0008333, -0.0010168, 0.0012057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0010260
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0010273
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938119, 0.9962257, 0.9937886, 0.9959753, -0.0014383, 0.0017306
1: -0.0028059, -0.0022044, -0.0028116, -0.0022668, -0.0003584, 0.0004312
2: 0.0016282, 0.0048157, 0.0019588, 0.0048462, -0.0022853, 0.0018992
3: -0.0034650, -0.0020142, -0.0034789, -0.0021647, -0.0008644, 0.0010402
4: 0.0008430, 0.0014600, 0.0009070, 0.0014659, -0.0004423, 0.0003676
5: 0.0010073, 0.0050163, 0.0014231, 0.0050547, -0.0028743, 0.0023887
6: 0.0002676, 0.0012852, 0.0002579, 0.0011796, -0.0006063, 0.0007295
7: -0.0024452, 0.0001875, -0.0024704, -0.0000856, -0.0015686, 0.0018875
8: -0.0008500, 0.0005345, -0.0008633, 0.0003909, -0.0008249, 0.0009926
9: -0.0024836, -0.0008782, -0.0023171, -0.0008628, -0.0011510, 0.0009565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009205, upper bound: 0.0009740
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009299, upper bound: 0.0009909
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937633, 0.9959769, -0.0014644, 0.0017548
1: -0.0028160, -0.0022102, -0.0028180, -0.0022664, -0.0003649, 0.0004373
2: 0.0016591, 0.0048696, 0.0019567, 0.0048799, -0.0023172, 0.0019338
3: -0.0034895, -0.0020283, -0.0034943, -0.0021638, -0.0008802, 0.0010547
4: 0.0008490, 0.0014704, 0.0009066, 0.0014724, -0.0004485, 0.0003743
5: 0.0010461, 0.0050841, 0.0014205, 0.0050971, -0.0029145, 0.0024322
6: 0.0002504, 0.0012753, 0.0002471, 0.0011803, -0.0006173, 0.0007397
7: -0.0024897, 0.0001620, -0.0024982, -0.0000839, -0.0015972, 0.0019139
8: -0.0008734, 0.0005210, -0.0008779, 0.0003917, -0.0008399, 0.0010065
9: -0.0024680, -0.0008511, -0.0023181, -0.0008458, -0.0011671, 0.0009739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009531, upper bound: 0.0010165
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009531, upper bound: 0.0010180
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938119, 0.9962257, 0.9937457, 0.9961591, -0.0016242, 0.0017648
1: -0.0028059, -0.0022044, -0.0028224, -0.0022210, -0.0004047, 0.0004397
2: 0.0016282, 0.0048157, 0.0017161, 0.0049030, -0.0023303, 0.0021447
3: -0.0034650, -0.0020142, -0.0035048, -0.0020542, -0.0009762, 0.0010607
4: 0.0008430, 0.0014600, 0.0008600, 0.0014769, -0.0004510, 0.0004151
5: 0.0010073, 0.0050163, 0.0011179, 0.0051262, -0.0029309, 0.0026975
6: 0.0002676, 0.0012852, 0.0002398, 0.0012571, -0.0006847, 0.0007439
7: -0.0024452, 0.0001875, -0.0025173, 0.0001149, -0.0017714, 0.0019247
8: -0.0008500, 0.0005345, -0.0008880, 0.0004963, -0.0009316, 0.0010122
9: -0.0024836, -0.0008782, -0.0024393, -0.0008342, -0.0011737, 0.0010802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009937, upper bound: 0.0009444
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0009613
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937202, 0.9961608, -0.0016581, 0.0017890
1: -0.0028160, -0.0022102, -0.0028287, -0.0022206, -0.0004132, 0.0004458
2: 0.0016591, 0.0048696, 0.0017139, 0.0049367, -0.0023624, 0.0021895
3: -0.0034895, -0.0020283, -0.0035201, -0.0020532, -0.0009966, 0.0010753
4: 0.0008490, 0.0014704, 0.0008596, 0.0014834, -0.0004572, 0.0004238
5: 0.0010461, 0.0050841, 0.0011151, 0.0051685, -0.0029713, 0.0027539
6: 0.0002504, 0.0012753, 0.0002290, 0.0012578, -0.0006990, 0.0007541
7: -0.0024897, 0.0001620, -0.0025451, 0.0001167, -0.0018084, 0.0019512
8: -0.0008734, 0.0005210, -0.0009026, 0.0004972, -0.0009510, 0.0010261
9: -0.0024680, -0.0008511, -0.0024404, -0.0008173, -0.0011898, 0.0011028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009870
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009882
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9937840, 0.9963966, 0.9937886, 0.9959753, -0.0014450, 0.0018896
1: -0.0028128, -0.0021618, -0.0028116, -0.0022668, -0.0003601, 0.0004708
2: 0.0014025, 0.0048524, 0.0019588, 0.0048462, -0.0024952, 0.0019081
3: -0.0034817, -0.0019115, -0.0034789, -0.0021647, -0.0008685, 0.0011357
4: 0.0007993, 0.0014671, 0.0009070, 0.0014659, -0.0004829, 0.0003693
5: 0.0007234, 0.0050625, 0.0014231, 0.0050547, -0.0031384, 0.0023999
6: 0.0002559, 0.0013572, 0.0002579, 0.0011796, -0.0006091, 0.0007965
7: -0.0024755, 0.0003740, -0.0024704, -0.0000856, -0.0015760, 0.0020609
8: -0.0008660, 0.0006325, -0.0008633, 0.0003909, -0.0008288, 0.0010838
9: -0.0025973, -0.0008597, -0.0023171, -0.0008628, -0.0012567, 0.0009610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008900, upper bound: 0.0010370
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008997, upper bound: 0.0010552
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9937633, 0.9959769, -0.0014707, 0.0019150
1: -0.0028227, -0.0021670, -0.0028180, -0.0022664, -0.0003665, 0.0004772
2: 0.0014297, 0.0049048, 0.0019567, 0.0048799, -0.0025287, 0.0019420
3: -0.0035056, -0.0019239, -0.0034943, -0.0021638, -0.0008839, 0.0011510
4: 0.0008046, 0.0014772, 0.0009066, 0.0014724, -0.0004894, 0.0003759
5: 0.0007576, 0.0051284, 0.0014205, 0.0050971, -0.0031805, 0.0024426
6: 0.0002392, 0.0013485, 0.0002471, 0.0011803, -0.0006200, 0.0008072
7: -0.0025188, 0.0003514, -0.0024982, -0.0000839, -0.0016040, 0.0020886
8: -0.0008887, 0.0006207, -0.0008779, 0.0003917, -0.0008435, 0.0010984
9: -0.0025836, -0.0008333, -0.0023181, -0.0008458, -0.0012736, 0.0009781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009211, upper bound: 0.0010795
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009211, upper bound: 0.0010801
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9937840, 0.9963966, 0.9937457, 0.9961591, -0.0015023, 0.0017897
1: -0.0028128, -0.0021618, -0.0028224, -0.0022210, -0.0003743, 0.0004460
2: 0.0014025, 0.0048524, 0.0017161, 0.0049030, -0.0023633, 0.0019837
3: -0.0034817, -0.0019115, -0.0035048, -0.0020542, -0.0009029, 0.0010757
4: 0.0007993, 0.0014671, 0.0008600, 0.0014769, -0.0004574, 0.0003840
5: 0.0007234, 0.0050625, 0.0011179, 0.0051262, -0.0029724, 0.0024950
6: 0.0002559, 0.0013572, 0.0002398, 0.0012571, -0.0006333, 0.0007544
7: -0.0024755, 0.0003740, -0.0025173, 0.0001149, -0.0016385, 0.0019520
8: -0.0008660, 0.0006325, -0.0008880, 0.0004963, -0.0008616, 0.0010265
9: -0.0025973, -0.0008597, -0.0024393, -0.0008342, -0.0011903, 0.0009991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008971, upper bound: 0.0010385
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010580
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9937202, 0.9961608, -0.0015289, 0.0018128
1: -0.0028227, -0.0021670, -0.0028287, -0.0022206, -0.0003810, 0.0004517
2: 0.0014297, 0.0049048, 0.0017139, 0.0049367, -0.0023938, 0.0020189
3: -0.0035056, -0.0019239, -0.0035201, -0.0020532, -0.0009189, 0.0010896
4: 0.0008046, 0.0014772, 0.0008596, 0.0014834, -0.0004633, 0.0003908
5: 0.0007576, 0.0051284, 0.0011151, 0.0051685, -0.0030108, 0.0025393
6: 0.0002392, 0.0013485, 0.0002290, 0.0012578, -0.0006445, 0.0007642
7: -0.0025188, 0.0003514, -0.0025451, 0.0001167, -0.0016675, 0.0019772
8: -0.0008887, 0.0006207, -0.0009026, 0.0004972, -0.0008769, 0.0010398
9: -0.0025836, -0.0008333, -0.0024404, -0.0008173, -0.0012057, 0.0010168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009276, upper bound: 0.0010819
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009276, upper bound: 0.0010827
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938119, 0.9962257, 0.9937546, 0.9962026, -0.0014171, 0.0014878
1: -0.0028059, -0.0022044, -0.0028201, -0.0022101, -0.0003531, 0.0003707
2: 0.0016282, 0.0048157, 0.0016586, 0.0048913, -0.0019646, 0.0018712
3: -0.0034650, -0.0020142, -0.0034994, -0.0020280, -0.0008517, 0.0008942
4: 0.0008430, 0.0014600, 0.0008489, 0.0014746, -0.0003802, 0.0003622
5: 0.0010073, 0.0050163, 0.0010455, 0.0051114, -0.0024710, 0.0023535
6: 0.0002676, 0.0012852, 0.0002435, 0.0012755, -0.0005973, 0.0006272
7: -0.0024452, 0.0001875, -0.0025076, 0.0001624, -0.0015455, 0.0016227
8: -0.0008500, 0.0005345, -0.0008829, 0.0005212, -0.0008128, 0.0008533
9: -0.0024836, -0.0008782, -0.0024683, -0.0008401, -0.0009895, 0.0009424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009161, upper bound: 0.0009846
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009337, upper bound: 0.0009923
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937322, 0.9962041, -0.0014494, 0.0015198
1: -0.0028160, -0.0022102, -0.0028257, -0.0022098, -0.0003612, 0.0003787
2: 0.0016591, 0.0048696, 0.0016567, 0.0049209, -0.0020069, 0.0019139
3: -0.0034895, -0.0020283, -0.0035129, -0.0020272, -0.0008711, 0.0009134
4: 0.0008490, 0.0014704, 0.0008485, 0.0014803, -0.0003884, 0.0003704
5: 0.0010461, 0.0050841, 0.0010431, 0.0051487, -0.0025241, 0.0024072
6: 0.0002504, 0.0012753, 0.0002340, 0.0012761, -0.0006110, 0.0006406
7: -0.0024897, 0.0001620, -0.0025321, 0.0001640, -0.0015808, 0.0016575
8: -0.0008734, 0.0005210, -0.0008958, 0.0005221, -0.0008313, 0.0008717
9: -0.0024680, -0.0008511, -0.0024692, -0.0008252, -0.0010108, 0.0009640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009563, upper bound: 0.0010177
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009563, upper bound: 0.0010188
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938119, 0.9962257, 0.9937276, 0.9963765, -0.0016022, 0.0015146
1: -0.0028059, -0.0022044, -0.0028269, -0.0021669, -0.0003992, 0.0003774
2: 0.0016282, 0.0048157, 0.0014292, 0.0049269, -0.0020000, 0.0021157
3: -0.0034650, -0.0020142, -0.0035156, -0.0019236, -0.0009630, 0.0009103
4: 0.0008430, 0.0014600, 0.0008045, 0.0014815, -0.0003871, 0.0004095
5: 0.0010073, 0.0050163, 0.0007570, 0.0051561, -0.0025154, 0.0026610
6: 0.0002676, 0.0012852, 0.0002321, 0.0013487, -0.0006754, 0.0006384
7: -0.0024452, 0.0001875, -0.0025370, 0.0003519, -0.0017474, 0.0016518
8: -0.0008500, 0.0005345, -0.0008983, 0.0006209, -0.0009190, 0.0008687
9: -0.0024836, -0.0008782, -0.0025838, -0.0008222, -0.0010073, 0.0010656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009846, upper bound: 0.0009556
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010037, upper bound: 0.0009619
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937045, 0.9963779, -0.0016417, 0.0015461
1: -0.0028160, -0.0022102, -0.0028326, -0.0021665, -0.0004091, 0.0003853
2: 0.0016591, 0.0048696, 0.0014272, 0.0049575, -0.0020416, 0.0021678
3: -0.0034895, -0.0020283, -0.0035296, -0.0019227, -0.0009867, 0.0009293
4: 0.0008490, 0.0014704, 0.0008041, 0.0014874, -0.0003952, 0.0004196
5: 0.0010461, 0.0050841, 0.0007545, 0.0051947, -0.0025678, 0.0027265
6: 0.0002504, 0.0012753, 0.0002224, 0.0013493, -0.0006920, 0.0006517
7: -0.0024897, 0.0001620, -0.0025623, 0.0003535, -0.0017905, 0.0016863
8: -0.0008734, 0.0005210, -0.0009116, 0.0006218, -0.0009416, 0.0008868
9: -0.0024680, -0.0008511, -0.0025848, -0.0008068, -0.0010283, 0.0010918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010264, upper bound: 0.0009874
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010264, upper bound: 0.0009884
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9937276, 0.9963765, 0.9938119, 0.9962257, -0.0015146, 0.0016022
1: -0.0028269, -0.0021669, -0.0028059, -0.0022044, -0.0003774, 0.0003992
2: 0.0014292, 0.0049269, 0.0016282, 0.0048157, -0.0021157, 0.0020000
3: -0.0035156, -0.0019236, -0.0034650, -0.0020142, -0.0009103, 0.0009630
4: 0.0008045, 0.0014815, 0.0008430, 0.0014600, -0.0004095, 0.0003871
5: 0.0007570, 0.0051561, 0.0010073, 0.0050163, -0.0026610, 0.0025154
6: 0.0002321, 0.0013487, 0.0002676, 0.0012852, -0.0006384, 0.0006754
7: -0.0025370, 0.0003519, -0.0024452, 0.0001875, -0.0016518, 0.0017474
8: -0.0008983, 0.0006209, -0.0008500, 0.0005345, -0.0008687, 0.0009190
9: -0.0025838, -0.0008222, -0.0024836, -0.0008782, -0.0010656, 0.0010073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008908, upper bound: 0.0010384
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010586
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9937045, 0.9963779, 0.9937711, 0.9962023, -0.0015461, 0.0016417
1: -0.0028326, -0.0021665, -0.0028160, -0.0022102, -0.0003853, 0.0004091
2: 0.0014272, 0.0049575, 0.0016591, 0.0048696, -0.0021678, 0.0020416
3: -0.0035296, -0.0019227, -0.0034895, -0.0020283, -0.0009293, 0.0009867
4: 0.0008041, 0.0014874, 0.0008490, 0.0014704, -0.0004196, 0.0003952
5: 0.0007545, 0.0051947, 0.0010461, 0.0050841, -0.0027265, 0.0025678
6: 0.0002224, 0.0013493, 0.0002504, 0.0012753, -0.0006517, 0.0006920
7: -0.0025623, 0.0003535, -0.0024897, 0.0001620, -0.0016863, 0.0017905
8: -0.0009116, 0.0006218, -0.0008734, 0.0005210, -0.0008868, 0.0009416
9: -0.0025848, -0.0008068, -0.0024680, -0.0008511, -0.0010918, 0.0010283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009231, upper bound: 0.0010801
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009231, upper bound: 0.0010806
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9937840, 0.9963966, 0.9937276, 0.9963765, -0.0014709, 0.0015421
1: -0.0028128, -0.0021618, -0.0028269, -0.0021669, -0.0003665, 0.0003843
2: 0.0014025, 0.0048524, 0.0014292, 0.0049269, -0.0020364, 0.0019423
3: -0.0034817, -0.0019115, -0.0035156, -0.0019236, -0.0008840, 0.0009269
4: 0.0007993, 0.0014671, 0.0008045, 0.0014815, -0.0003941, 0.0003759
5: 0.0007234, 0.0050625, 0.0007570, 0.0051561, -0.0025612, 0.0024429
6: 0.0002559, 0.0013572, 0.0002321, 0.0013487, -0.0006200, 0.0006501
7: -0.0024755, 0.0003740, -0.0025370, 0.0003519, -0.0016042, 0.0016819
8: -0.0008660, 0.0006325, -0.0008983, 0.0006209, -0.0008436, 0.0008845
9: -0.0025973, -0.0008597, -0.0025838, -0.0008222, -0.0010256, 0.0009782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008949, upper bound: 0.0010493
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009101, upper bound: 0.0010588
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9937045, 0.9963779, -0.0015034, 0.0015719
1: -0.0028227, -0.0021670, -0.0028326, -0.0021665, -0.0003746, 0.0003917
2: 0.0014297, 0.0049048, 0.0014272, 0.0049575, -0.0020757, 0.0019852
3: -0.0035056, -0.0019239, -0.0035296, -0.0019227, -0.0009036, 0.0009448
4: 0.0008046, 0.0014772, 0.0008041, 0.0014874, -0.0004017, 0.0003842
5: 0.0007576, 0.0051284, 0.0007545, 0.0051947, -0.0026107, 0.0024969
6: 0.0002392, 0.0013485, 0.0002224, 0.0013493, -0.0006337, 0.0006626
7: -0.0025188, 0.0003514, -0.0025623, 0.0003535, -0.0016397, 0.0017144
8: -0.0008887, 0.0006207, -0.0009116, 0.0006218, -0.0008623, 0.0009016
9: -0.0025836, -0.0008333, -0.0025848, -0.0008068, -0.0010454, 0.0009999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009303, upper bound: 0.0010825
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009303, upper bound: 0.0010834
time: 0.77 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009195
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009304, upper bound: 0.0009288
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009533, upper bound: 0.0009533
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009533, upper bound: 0.0009562
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009831, upper bound: 0.0008893
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0008982
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009225
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009240
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008893, upper bound: 0.0009832
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010018
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009225, upper bound: 0.0010244
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009225, upper bound: 0.0010254
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008979, upper bound: 0.0009840
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0010021
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009289, upper bound: 0.0010261
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009289, upper bound: 0.0010273
IS_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009740, upper bound: 0.0009206
IS_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009909, upper bound: 0.0009300
IS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010165, upper bound: 0.0009532
IS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010165, upper bound: 0.0009559
IS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009444, upper bound: 0.0009937
IS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009613, upper bound: 0.0010018
IS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010244
IS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009870, upper bound: 0.0010254
IS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010370, upper bound: 0.0008900
IS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008997
IS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010795, upper bound: 0.0009211
IS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010795, upper bound: 0.0009226
IS_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009504, upper bound: 0.0009949
IS_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009681, upper bound: 0.0010036
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0010260
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009930, upper bound: 0.0010273
IS_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009205, upper bound: 0.0009740
IS_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009299, upper bound: 0.0009909
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009531, upper bound: 0.0010165
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009531, upper bound: 0.0010180
IS_A2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009937, upper bound: 0.0009444
IS_A2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0009613
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009870
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010244, upper bound: 0.0009882
IS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008900, upper bound: 0.0010370
IS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008997, upper bound: 0.0010552
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009211, upper bound: 0.0010795
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009211, upper bound: 0.0010801
IS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008971, upper bound: 0.0010385
IS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009074, upper bound: 0.0010580
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009276, upper bound: 0.0010819
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009276, upper bound: 0.0010827
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009161, upper bound: 0.0009846
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009337, upper bound: 0.0009923
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009563, upper bound: 0.0010177
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009563, upper bound: 0.0010188
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009846, upper bound: 0.0009556
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010037, upper bound: 0.0009619
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010264, upper bound: 0.0009874
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0010264, upper bound: 0.0009884
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008908, upper bound: 0.0010384
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010586
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009231, upper bound: 0.0010801
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009231, upper bound: 0.0010806
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0008949, upper bound: 0.0010493
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009101, upper bound: 0.0010588
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009303, upper bound: 0.0010825
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0009303, upper bound: 0.0010834

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938651, 0.9959879, 0.9938510, 0.9959909, -0.0012848, 0.0012846
1: -0.0027926, -0.0022637, -0.0027961, -0.0022629, -0.0003201, 0.0003201
2: 0.0019422, 0.0047453, 0.0019384, 0.0047641, -0.0016963, 0.0016966
3: -0.0034330, -0.0021571, -0.0034415, -0.0021554, -0.0007722, 0.0007721
4: 0.0009038, 0.0014463, 0.0009031, 0.0014500, -0.0003283, 0.0003284
5: 0.0014022, 0.0049278, 0.0013974, 0.0049514, -0.0021335, 0.0021339
6: 0.0002901, 0.0011849, 0.0002841, 0.0011862, -0.0005416, 0.0005415
7: -0.0023871, -0.0000718, -0.0024025, -0.0000687, -0.0014013, 0.0014010
8: -0.0008195, 0.0003981, -0.0008276, 0.0003997, -0.0007369, 0.0007368
9: -0.0023254, -0.0009136, -0.0023274, -0.0009042, -0.0008543, 0.0008545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009191
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009194
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938492, 0.9959928, 0.9938241, 0.9959716, -0.0012958, 0.0013298
1: -0.0027966, -0.0022625, -0.0028028, -0.0022677, -0.0003229, 0.0003313
2: 0.0019359, 0.0047663, 0.0019638, 0.0047995, -0.0017559, 0.0017111
3: -0.0034426, -0.0021542, -0.0034576, -0.0021670, -0.0007788, 0.0007992
4: 0.0009026, 0.0014504, 0.0009080, 0.0014568, -0.0003399, 0.0003312
5: 0.0013942, 0.0049543, 0.0014294, 0.0049959, -0.0022085, 0.0021521
6: 0.0002834, 0.0011870, 0.0002728, 0.0011780, -0.0005462, 0.0005605
7: -0.0024044, -0.0000666, -0.0024318, -0.0000897, -0.0014133, 0.0014503
8: -0.0008286, 0.0004008, -0.0008430, 0.0003887, -0.0007432, 0.0007627
9: -0.0023286, -0.0009030, -0.0023146, -0.0008864, -0.0008844, 0.0008618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0009287
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0009288
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9938462, 0.9959931, -0.0013619, 0.0013011
1: -0.0028080, -0.0022669, -0.0027973, -0.0022624, -0.0003393, 0.0003242
2: 0.0019593, 0.0048270, 0.0019353, 0.0047704, -0.0017180, 0.0017983
3: -0.0034702, -0.0021649, -0.0034444, -0.0021540, -0.0008185, 0.0007820
4: 0.0009071, 0.0014621, 0.0009025, 0.0014512, -0.0003325, 0.0003481
5: 0.0014237, 0.0050306, 0.0013936, 0.0049593, -0.0021608, 0.0022618
6: 0.0002640, 0.0011795, 0.0002821, 0.0011871, -0.0005741, 0.0005484
7: -0.0024545, -0.0000860, -0.0024077, -0.0000662, -0.0014853, 0.0014190
8: -0.0008550, 0.0003907, -0.0008303, 0.0004010, -0.0007811, 0.0007462
9: -0.0023168, -0.0008725, -0.0023289, -0.0009010, -0.0008653, 0.0009057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009195, upper bound: 0.0009113
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009287, upper bound: 0.0009288
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9938033, 0.9959750, -0.0013289, 0.0013289
1: -0.0028080, -0.0022669, -0.0028080, -0.0022669, -0.0003311, 0.0003311
2: 0.0019593, 0.0048270, 0.0019593, 0.0048270, -0.0017548, 0.0017548
3: -0.0034702, -0.0021649, -0.0034702, -0.0021649, -0.0007987, 0.0007987
4: 0.0009071, 0.0014621, 0.0009071, 0.0014621, -0.0003396, 0.0003396
5: 0.0014237, 0.0050306, 0.0014237, 0.0050306, -0.0022071, 0.0022071
6: 0.0002640, 0.0011795, 0.0002640, 0.0011795, -0.0005602, 0.0005602
7: -0.0024545, -0.0000860, -0.0024545, -0.0000860, -0.0014494, 0.0014494
8: -0.0008550, 0.0003907, -0.0008550, 0.0003907, -0.0007622, 0.0007622
9: -0.0023168, -0.0008725, -0.0023168, -0.0008725, -0.0008838, 0.0008838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009195, upper bound: 0.0009159
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009287, upper bound: 0.0009315
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938651, 0.9959879, 0.9938031, 0.9961760, -0.0014781, 0.0013306
1: -0.0027926, -0.0022637, -0.0028080, -0.0022168, -0.0003683, 0.0003315
2: 0.0019422, 0.0047453, 0.0016938, 0.0048272, -0.0017570, 0.0019518
3: -0.0034330, -0.0021571, -0.0034702, -0.0020441, -0.0008884, 0.0007997
4: 0.0009038, 0.0014463, 0.0008557, 0.0014622, -0.0003401, 0.0003778
5: 0.0014022, 0.0049278, 0.0010898, 0.0050307, -0.0022098, 0.0024549
6: 0.0002901, 0.0011849, 0.0002640, 0.0012642, -0.0006231, 0.0005609
7: -0.0023871, -0.0000718, -0.0024546, 0.0001333, -0.0016121, 0.0014512
8: -0.0008195, 0.0003981, -0.0008550, 0.0005059, -0.0008478, 0.0007631
9: -0.0023254, -0.0009136, -0.0024505, -0.0008724, -0.0008849, 0.0009830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009831, upper bound: 0.0008892
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009831, upper bound: 0.0008893
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938492, 0.9959928, 0.9937817, 0.9961553, -0.0014813, 0.0013656
1: -0.0027966, -0.0022625, -0.0028134, -0.0022220, -0.0003691, 0.0003403
2: 0.0019359, 0.0047663, 0.0017212, 0.0048555, -0.0018032, 0.0019561
3: -0.0034426, -0.0021542, -0.0034831, -0.0020565, -0.0008903, 0.0008208
4: 0.0009026, 0.0014504, 0.0008610, 0.0014677, -0.0003490, 0.0003786
5: 0.0013942, 0.0049543, 0.0011243, 0.0050664, -0.0022680, 0.0024602
6: 0.0002834, 0.0011870, 0.0002549, 0.0012555, -0.0006244, 0.0005756
7: -0.0024044, -0.0000666, -0.0024781, 0.0001107, -0.0016156, 0.0014894
8: -0.0008286, 0.0004008, -0.0008673, 0.0004941, -0.0008496, 0.0007832
9: -0.0023286, -0.0009030, -0.0024367, -0.0008581, -0.0009082, 0.0009852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0008982
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0008982
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9938022, 0.9961815, -0.0015561, 0.0013420
1: -0.0028080, -0.0022669, -0.0028083, -0.0022154, -0.0003877, 0.0003344
2: 0.0019593, 0.0048270, 0.0016866, 0.0048283, -0.0017721, 0.0020549
3: -0.0034702, -0.0021649, -0.0034708, -0.0020408, -0.0009353, 0.0008066
4: 0.0009071, 0.0014621, 0.0008543, 0.0014624, -0.0003430, 0.0003977
5: 0.0014237, 0.0050306, 0.0010807, 0.0050322, -0.0022289, 0.0025845
6: 0.0002640, 0.0011795, 0.0002636, 0.0012665, -0.0006560, 0.0005657
7: -0.0024545, -0.0000860, -0.0024556, 0.0001393, -0.0016972, 0.0014637
8: -0.0008550, 0.0003907, -0.0008555, 0.0005091, -0.0008925, 0.0007697
9: -0.0023168, -0.0008725, -0.0024542, -0.0008718, -0.0008925, 0.0010349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0008838
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0008983
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9937605, 0.9961587, -0.0015223, 0.0013633
1: -0.0028080, -0.0022669, -0.0028187, -0.0022211, -0.0003793, 0.0003397
2: 0.0019593, 0.0048270, 0.0017166, 0.0048835, -0.0018002, 0.0020102
3: -0.0034702, -0.0021649, -0.0034959, -0.0020545, -0.0009150, 0.0008194
4: 0.0009071, 0.0014621, 0.0008601, 0.0014731, -0.0003484, 0.0003891
5: 0.0014237, 0.0050306, 0.0011185, 0.0051016, -0.0022641, 0.0025283
6: 0.0002640, 0.0011795, 0.0002460, 0.0012569, -0.0006417, 0.0005747
7: -0.0024545, -0.0000860, -0.0025011, 0.0001145, -0.0016603, 0.0014868
8: -0.0008550, 0.0003907, -0.0008795, 0.0004961, -0.0008731, 0.0007819
9: -0.0023168, -0.0008725, -0.0024391, -0.0008441, -0.0009067, 0.0010125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0008862
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0008999
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938031, 0.9961760, 0.9938651, 0.9959879, -0.0013306, 0.0014781
1: -0.0028080, -0.0022168, -0.0027926, -0.0022637, -0.0003315, 0.0003683
2: 0.0016938, 0.0048272, 0.0019422, 0.0047453, -0.0019518, 0.0017570
3: -0.0034702, -0.0020441, -0.0034330, -0.0021571, -0.0007997, 0.0008884
4: 0.0008557, 0.0014622, 0.0009038, 0.0014463, -0.0003778, 0.0003401
5: 0.0010898, 0.0050307, 0.0014022, 0.0049278, -0.0024549, 0.0022098
6: 0.0002640, 0.0012642, 0.0002901, 0.0011849, -0.0005609, 0.0006231
7: -0.0024546, 0.0001333, -0.0023871, -0.0000718, -0.0014512, 0.0016121
8: -0.0008550, 0.0005059, -0.0008195, 0.0003981, -0.0007631, 0.0008478
9: -0.0024505, -0.0008724, -0.0023254, -0.0009136, -0.0009830, 0.0008849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009831
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009831
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937817, 0.9961553, 0.9938492, 0.9959928, -0.0013656, 0.0014813
1: -0.0028134, -0.0022220, -0.0027966, -0.0022625, -0.0003403, 0.0003691
2: 0.0017212, 0.0048555, 0.0019359, 0.0047663, -0.0019561, 0.0018032
3: -0.0034831, -0.0020565, -0.0034426, -0.0021542, -0.0008208, 0.0008903
4: 0.0008610, 0.0014677, 0.0009026, 0.0014504, -0.0003786, 0.0003490
5: 0.0011243, 0.0050664, 0.0013942, 0.0049543, -0.0024602, 0.0022680
6: 0.0002549, 0.0012555, 0.0002834, 0.0011870, -0.0005756, 0.0006244
7: -0.0024781, 0.0001107, -0.0024044, -0.0000666, -0.0014894, 0.0016156
8: -0.0008673, 0.0004941, -0.0008286, 0.0004008, -0.0007832, 0.0008496
9: -0.0024367, -0.0008581, -0.0023286, -0.0009030, -0.0009852, 0.0009082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010018
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010018
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938022, 0.9961815, 0.9938033, 0.9959750, -0.0013420, 0.0015561
1: -0.0028083, -0.0022154, -0.0028080, -0.0022669, -0.0003344, 0.0003877
2: 0.0016866, 0.0048283, 0.0019593, 0.0048270, -0.0020549, 0.0017721
3: -0.0034708, -0.0020408, -0.0034702, -0.0021649, -0.0008066, 0.0009353
4: 0.0008543, 0.0014624, 0.0009071, 0.0014621, -0.0003977, 0.0003430
5: 0.0010807, 0.0050322, 0.0014237, 0.0050306, -0.0025845, 0.0022289
6: 0.0002636, 0.0012665, 0.0002640, 0.0011795, -0.0005657, 0.0006560
7: -0.0024556, 0.0001393, -0.0024545, -0.0000860, -0.0014637, 0.0016972
8: -0.0008555, 0.0005091, -0.0008550, 0.0003907, -0.0007697, 0.0008925
9: -0.0024542, -0.0008718, -0.0023168, -0.0008725, -0.0010349, 0.0008925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008837, upper bound: 0.0009915
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010002
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937605, 0.9961587, 0.9938033, 0.9959750, -0.0013633, 0.0015223
1: -0.0028187, -0.0022211, -0.0028080, -0.0022669, -0.0003397, 0.0003793
2: 0.0017166, 0.0048835, 0.0019593, 0.0048270, -0.0020102, 0.0018002
3: -0.0034959, -0.0020545, -0.0034702, -0.0021649, -0.0008194, 0.0009150
4: 0.0008601, 0.0014731, 0.0009071, 0.0014621, -0.0003891, 0.0003484
5: 0.0011185, 0.0051016, 0.0014237, 0.0050306, -0.0025283, 0.0022641
6: 0.0002460, 0.0012569, 0.0002640, 0.0011795, -0.0005747, 0.0006417
7: -0.0025011, 0.0001145, -0.0024545, -0.0000860, -0.0014868, 0.0016603
8: -0.0008795, 0.0004961, -0.0008550, 0.0003907, -0.0007819, 0.0008731
9: -0.0024391, -0.0008441, -0.0023168, -0.0008725, -0.0010125, 0.0009067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008837, upper bound: 0.0009935
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010013
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9938560, 0.9961986, 0.9937667, 0.9961540, -0.0012938, 0.0014087
1: -0.0027949, -0.0022112, -0.0028171, -0.0022223, -0.0003224, 0.0003510
2: 0.0016640, 0.0047573, 0.0017229, 0.0048754, -0.0018602, 0.0017085
3: -0.0034385, -0.0020305, -0.0034922, -0.0020573, -0.0007776, 0.0008467
4: 0.0008500, 0.0014487, 0.0008614, 0.0014715, -0.0003600, 0.0003307
5: 0.0010523, 0.0049429, 0.0011265, 0.0050914, -0.0023397, 0.0021488
6: 0.0002863, 0.0012737, 0.0002486, 0.0012549, -0.0005454, 0.0005938
7: -0.0023970, 0.0001579, -0.0024945, 0.0001093, -0.0014111, 0.0015364
8: -0.0008247, 0.0005189, -0.0008760, 0.0004933, -0.0007421, 0.0008080
9: -0.0024656, -0.0009076, -0.0024359, -0.0008481, -0.0009369, 0.0008605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008979, upper bound: 0.0009839
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008979, upper bound: 0.0009840
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9938353, 0.9961774, 0.9937491, 0.9961587, -0.0013282, 0.0014210
1: -0.0028001, -0.0022165, -0.0028215, -0.0022211, -0.0003310, 0.0003541
2: 0.0016921, 0.0047848, 0.0017166, 0.0048984, -0.0018765, 0.0017539
3: -0.0034510, -0.0020433, -0.0035027, -0.0020545, -0.0007983, 0.0008541
4: 0.0008554, 0.0014540, 0.0008601, 0.0014760, -0.0003632, 0.0003395
5: 0.0010877, 0.0049775, 0.0011185, 0.0051204, -0.0023601, 0.0022059
6: 0.0002775, 0.0012648, 0.0002412, 0.0012569, -0.0005599, 0.0005990
7: -0.0024197, 0.0001347, -0.0025135, 0.0001145, -0.0014486, 0.0015498
8: -0.0008366, 0.0005067, -0.0008860, 0.0004960, -0.0007618, 0.0008150
9: -0.0024514, -0.0008937, -0.0024390, -0.0008365, -0.0009451, 0.0008833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0010021
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0010021
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937605, 0.9961587, 0.9938022, 0.9961815, -0.0014228, 0.0013632
1: -0.0028187, -0.0022211, -0.0028083, -0.0022154, -0.0003545, 0.0003397
2: 0.0017166, 0.0048835, 0.0016866, 0.0048283, -0.0018001, 0.0018788
3: -0.0034959, -0.0020545, -0.0034708, -0.0020408, -0.0008552, 0.0008193
4: 0.0008601, 0.0014731, 0.0008543, 0.0014624, -0.0003484, 0.0003636
5: 0.0011185, 0.0051016, 0.0010807, 0.0050322, -0.0022640, 0.0023631
6: 0.0002460, 0.0012569, 0.0002636, 0.0012665, -0.0005998, 0.0005746
7: -0.0025011, 0.0001145, -0.0024556, 0.0001393, -0.0015518, 0.0014867
8: -0.0008795, 0.0004961, -0.0008555, 0.0005091, -0.0008161, 0.0007819
9: -0.0024391, -0.0008441, -0.0024542, -0.0008718, -0.0009066, 0.0009463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008902, upper bound: 0.0009929
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009052, upper bound: 0.0010021
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937605, 0.9961587, 0.9937605, 0.9961587, -0.0013894, 0.0013894
1: -0.0028187, -0.0022211, -0.0028187, -0.0022211, -0.0003462, 0.0003462
2: 0.0017166, 0.0048835, 0.0017166, 0.0048835, -0.0018347, 0.0018347
3: -0.0034959, -0.0020545, -0.0034959, -0.0020545, -0.0008351, 0.0008351
4: 0.0008601, 0.0014731, 0.0008601, 0.0014731, -0.0003551, 0.0003551
5: 0.0011185, 0.0051016, 0.0011185, 0.0051016, -0.0023076, 0.0023076
6: 0.0002460, 0.0012569, 0.0002460, 0.0012569, -0.0005857, 0.0005857
7: -0.0025011, 0.0001145, -0.0025011, 0.0001145, -0.0015154, 0.0015154
8: -0.0008795, 0.0004961, -0.0008795, 0.0004961, -0.0007969, 0.0007969
9: -0.0024391, -0.0008441, -0.0024391, -0.0008441, -0.0009241, 0.0009241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008959, upper bound: 0.0009854
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009052, upper bound: 0.0010037
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9938102, 0.9959704, 0.9938645, 0.9962469, -0.0017250, 0.0013811
1: -0.0028063, -0.0022680, -0.0027928, -0.0021991, -0.0004298, 0.0003441
2: 0.0019653, 0.0048179, 0.0016001, 0.0047462, -0.0018237, 0.0022778
3: -0.0034660, -0.0021676, -0.0034334, -0.0020014, -0.0010368, 0.0008301
4: 0.0009083, 0.0014604, 0.0008376, 0.0014465, -0.0003530, 0.0004409
5: 0.0014313, 0.0050191, 0.0009720, 0.0049289, -0.0022938, 0.0028649
6: 0.0002669, 0.0011776, 0.0002898, 0.0012941, -0.0007271, 0.0005822
7: -0.0024470, -0.0000909, -0.0023878, 0.0002107, -0.0018814, 0.0015063
8: -0.0008510, 0.0003880, -0.0008199, 0.0005467, -0.0009894, 0.0007921
9: -0.0023138, -0.0008771, -0.0024977, -0.0009132, -0.0009185, 0.0011472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009206
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009206
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9937921, 0.9959750, 0.9938465, 0.9962219, -0.0017248, 0.0014050
1: -0.0028108, -0.0022669, -0.0027972, -0.0022054, -0.0004298, 0.0003501
2: 0.0019593, 0.0048418, 0.0016333, 0.0047699, -0.0018553, 0.0022776
3: -0.0034769, -0.0021649, -0.0034442, -0.0020165, -0.0010367, 0.0008444
4: 0.0009071, 0.0014650, 0.0008440, 0.0014511, -0.0003591, 0.0004408
5: 0.0014237, 0.0050492, 0.0010137, 0.0049587, -0.0023334, 0.0028646
6: 0.0002593, 0.0011795, 0.0002823, 0.0012835, -0.0007271, 0.0005923
7: -0.0024667, -0.0000860, -0.0024073, 0.0001833, -0.0018811, 0.0015323
8: -0.0008614, 0.0003906, -0.0008301, 0.0005322, -0.0009893, 0.0008058
9: -0.0023168, -0.0008650, -0.0024810, -0.0009013, -0.0009344, 0.0011471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009300
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009300
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938462, 0.9959931, 0.9937711, 0.9962023, -0.0016567, 0.0014914
1: -0.0027973, -0.0022624, -0.0028160, -0.0022102, -0.0004128, 0.0003716
2: 0.0019353, 0.0047704, 0.0016591, 0.0048696, -0.0019694, 0.0021876
3: -0.0034444, -0.0021540, -0.0034895, -0.0020283, -0.0009957, 0.0008964
4: 0.0009025, 0.0014512, 0.0008490, 0.0014704, -0.0003812, 0.0004234
5: 0.0013936, 0.0049593, 0.0010461, 0.0050841, -0.0024770, 0.0027515
6: 0.0002821, 0.0011871, 0.0002504, 0.0012753, -0.0006984, 0.0006287
7: -0.0024077, -0.0000662, -0.0024897, 0.0001620, -0.0018069, 0.0016266
8: -0.0008303, 0.0004010, -0.0008734, 0.0005210, -0.0009502, 0.0008554
9: -0.0023289, -0.0009010, -0.0024680, -0.0008511, -0.0009919, 0.0011018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009191
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009285
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9937711, 0.9962023, -0.0016914, 0.0014627
1: -0.0028080, -0.0022669, -0.0028160, -0.0022102, -0.0004214, 0.0003645
2: 0.0019593, 0.0048270, 0.0016591, 0.0048696, -0.0019314, 0.0022334
3: -0.0034702, -0.0021649, -0.0034895, -0.0020283, -0.0010166, 0.0008791
4: 0.0009071, 0.0014621, 0.0008490, 0.0014704, -0.0003738, 0.0004323
5: 0.0014237, 0.0050306, 0.0010461, 0.0050841, -0.0024292, 0.0028091
6: 0.0002640, 0.0011795, 0.0002504, 0.0012753, -0.0007130, 0.0006166
7: -0.0024545, -0.0000860, -0.0024897, 0.0001620, -0.0018447, 0.0015952
8: -0.0008550, 0.0003907, -0.0008734, 0.0005210, -0.0009701, 0.0008389
9: -0.0023168, -0.0008725, -0.0024680, -0.0008511, -0.0009728, 0.0011249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009231
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009310
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9937667, 0.9961540, 0.9938645, 0.9962469, -0.0017621, 0.0015663
1: -0.0028171, -0.0022223, -0.0027928, -0.0021991, -0.0004391, 0.0003903
2: 0.0017229, 0.0048754, 0.0016001, 0.0047462, -0.0020683, 0.0023269
3: -0.0034922, -0.0020573, -0.0034334, -0.0020014, -0.0010591, 0.0009414
4: 0.0008614, 0.0014715, 0.0008376, 0.0014465, -0.0004003, 0.0004504
5: 0.0011265, 0.0050914, 0.0009720, 0.0049289, -0.0026014, 0.0029266
6: 0.0002486, 0.0012549, 0.0002898, 0.0012941, -0.0007428, 0.0006603
7: -0.0024945, 0.0001093, -0.0023878, 0.0002107, -0.0019219, 0.0017083
8: -0.0008760, 0.0004933, -0.0008199, 0.0005467, -0.0010107, 0.0008984
9: -0.0024359, -0.0008481, -0.0024977, -0.0009132, -0.0010417, 0.0011719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009938
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009938
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9937491, 0.9961587, 0.9938465, 0.9962219, -0.0017586, 0.0015957
1: -0.0028215, -0.0022211, -0.0027972, -0.0022054, -0.0004382, 0.0003976
2: 0.0017166, 0.0048984, 0.0016333, 0.0047699, -0.0021071, 0.0023222
3: -0.0035027, -0.0020545, -0.0034442, -0.0020165, -0.0010570, 0.0009591
4: 0.0008601, 0.0014760, 0.0008440, 0.0014511, -0.0004078, 0.0004495
5: 0.0011185, 0.0051204, 0.0010137, 0.0049587, -0.0026502, 0.0029208
6: 0.0002412, 0.0012569, 0.0002823, 0.0012835, -0.0007413, 0.0006726
7: -0.0025135, 0.0001145, -0.0024073, 0.0001833, -0.0019180, 0.0017403
8: -0.0008860, 0.0004960, -0.0008301, 0.0005322, -0.0010087, 0.0009152
9: -0.0024390, -0.0008365, -0.0024810, -0.0009013, -0.0010613, 0.0011696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010018
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010018
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938022, 0.9961815, 0.9937711, 0.9962023, -0.0016977, 0.0016857
1: -0.0028083, -0.0022154, -0.0028160, -0.0022102, -0.0004230, 0.0004200
2: 0.0016866, 0.0048283, 0.0016591, 0.0048696, -0.0022259, 0.0022417
3: -0.0034708, -0.0020408, -0.0034895, -0.0020283, -0.0010203, 0.0010131
4: 0.0008543, 0.0014624, 0.0008490, 0.0014704, -0.0004308, 0.0004339
5: 0.0010807, 0.0050322, 0.0010461, 0.0050841, -0.0027996, 0.0028195
6: 0.0002636, 0.0012665, 0.0002504, 0.0012753, -0.0007156, 0.0007106
7: -0.0024556, 0.0001393, -0.0024897, 0.0001620, -0.0018515, 0.0018385
8: -0.0008555, 0.0005091, -0.0008734, 0.0005210, -0.0009737, 0.0009668
9: -0.0024542, -0.0008718, -0.0024680, -0.0008511, -0.0011211, 0.0011291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009915
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010002
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937605, 0.9961587, 0.9937711, 0.9962023, -0.0017257, 0.0016561
1: -0.0028187, -0.0022211, -0.0028160, -0.0022102, -0.0004300, 0.0004126
2: 0.0017166, 0.0048835, 0.0016591, 0.0048696, -0.0021868, 0.0022788
3: -0.0034959, -0.0020545, -0.0034895, -0.0020283, -0.0010372, 0.0009953
4: 0.0008601, 0.0014731, 0.0008490, 0.0014704, -0.0004233, 0.0004411
5: 0.0011185, 0.0051016, 0.0010461, 0.0050841, -0.0027504, 0.0028661
6: 0.0002460, 0.0012569, 0.0002504, 0.0012753, -0.0007275, 0.0006981
7: -0.0025011, 0.0001145, -0.0024897, 0.0001620, -0.0018821, 0.0018062
8: -0.0008795, 0.0004961, -0.0008734, 0.0005210, -0.0009898, 0.0009499
9: -0.0024391, -0.0008441, -0.0024680, -0.0008511, -0.0011014, 0.0011477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009935
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010013
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9938102, 0.9959704, 0.9938354, 0.9964168, -0.0018832, 0.0013936
1: -0.0028063, -0.0022680, -0.0028000, -0.0021568, -0.0004692, 0.0003472
2: 0.0019653, 0.0048179, 0.0013760, 0.0047845, -0.0018402, 0.0024868
3: -0.0034660, -0.0021676, -0.0034508, -0.0018994, -0.0011319, 0.0008376
4: 0.0009083, 0.0014604, 0.0007942, 0.0014539, -0.0003562, 0.0004813
5: 0.0014313, 0.0050191, 0.0006900, 0.0049771, -0.0023145, 0.0031277
6: 0.0002669, 0.0011776, 0.0002776, 0.0013657, -0.0007938, 0.0005874
7: -0.0024470, -0.0000909, -0.0024194, 0.0003958, -0.0020539, 0.0015199
8: -0.0008510, 0.0003880, -0.0008365, 0.0006440, -0.0010801, 0.0007993
9: -0.0023138, -0.0008771, -0.0026106, -0.0008939, -0.0009268, 0.0012525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008900
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008900
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9937921, 0.9959750, 0.9938160, 0.9963927, -0.0018831, 0.0014134
1: -0.0028108, -0.0022669, -0.0028048, -0.0021628, -0.0004692, 0.0003522
2: 0.0019593, 0.0048418, 0.0014076, 0.0048102, -0.0018664, 0.0024867
3: -0.0034769, -0.0021649, -0.0034625, -0.0019138, -0.0011318, 0.0008495
4: 0.0009071, 0.0014650, 0.0008003, 0.0014589, -0.0003612, 0.0004813
5: 0.0014237, 0.0050492, 0.0007299, 0.0050094, -0.0023474, 0.0031276
6: 0.0002593, 0.0011795, 0.0002694, 0.0013556, -0.0007938, 0.0005958
7: -0.0024667, -0.0000860, -0.0024406, 0.0003697, -0.0020538, 0.0015415
8: -0.0008614, 0.0003906, -0.0008477, 0.0006303, -0.0010801, 0.0008107
9: -0.0023168, -0.0008650, -0.0025947, -0.0008810, -0.0009400, 0.0012524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008997
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008997
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938462, 0.9959931, 0.9937444, 0.9963761, -0.0018168, 0.0015026
1: -0.0027973, -0.0022624, -0.0028227, -0.0021670, -0.0004527, 0.0003744
2: 0.0019353, 0.0047704, 0.0014297, 0.0049048, -0.0019841, 0.0023991
3: -0.0034444, -0.0021540, -0.0035056, -0.0019239, -0.0010920, 0.0009031
4: 0.0009025, 0.0014512, 0.0008046, 0.0014772, -0.0003840, 0.0004643
5: 0.0013936, 0.0049593, 0.0007576, 0.0051284, -0.0024955, 0.0030175
6: 0.0002821, 0.0011871, 0.0002392, 0.0013485, -0.0007659, 0.0006334
7: -0.0024077, -0.0000662, -0.0025188, 0.0003514, -0.0019815, 0.0016388
8: -0.0008303, 0.0004010, -0.0008887, 0.0006207, -0.0010421, 0.0008618
9: -0.0023289, -0.0009010, -0.0025836, -0.0008333, -0.0009993, 0.0012083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008877
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008967
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938033, 0.9959750, 0.9937444, 0.9963761, -0.0018504, 0.0014689
1: -0.0028080, -0.0022669, -0.0028227, -0.0021670, -0.0004611, 0.0003660
2: 0.0019593, 0.0048270, 0.0014297, 0.0049048, -0.0019397, 0.0024435
3: -0.0034702, -0.0021649, -0.0035056, -0.0019239, -0.0011122, 0.0008829
4: 0.0009071, 0.0014621, 0.0008046, 0.0014772, -0.0003754, 0.0004729
5: 0.0014237, 0.0050306, 0.0007576, 0.0051284, -0.0024397, 0.0030732
6: 0.0002640, 0.0011795, 0.0002392, 0.0013485, -0.0007800, 0.0006192
7: -0.0024545, -0.0000860, -0.0025188, 0.0003514, -0.0020181, 0.0016021
8: -0.0008550, 0.0003907, -0.0008887, 0.0006207, -0.0010613, 0.0008425
9: -0.0023168, -0.0008725, -0.0025836, -0.0008333, -0.0009769, 0.0012307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008906
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008988
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9937667, 0.9961540, 0.9938354, 0.9964168, -0.0017855, 0.0014478
1: -0.0028171, -0.0022223, -0.0028000, -0.0021568, -0.0004449, 0.0003608
2: 0.0017229, 0.0048754, 0.0013760, 0.0047845, -0.0019118, 0.0023577
3: -0.0034922, -0.0020573, -0.0034508, -0.0018994, -0.0010731, 0.0008702
4: 0.0008614, 0.0014715, 0.0007942, 0.0014539, -0.0003700, 0.0004563
5: 0.0011265, 0.0050914, 0.0006900, 0.0049771, -0.0024046, 0.0029654
6: 0.0002486, 0.0012549, 0.0002776, 0.0013657, -0.0007526, 0.0006103
7: -0.0024945, 0.0001093, -0.0024194, 0.0003958, -0.0019473, 0.0015791
8: -0.0008760, 0.0004933, -0.0008365, 0.0006440, -0.0010241, 0.0008304
9: -0.0024359, -0.0008481, -0.0026106, -0.0008939, -0.0009629, 0.0011875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009950
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009950
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9937491, 0.9961587, 0.9938160, 0.9963927, -0.0017836, 0.0014705
1: -0.0028215, -0.0022211, -0.0028048, -0.0021628, -0.0004444, 0.0003664
2: 0.0017166, 0.0048984, 0.0014076, 0.0048102, -0.0019417, 0.0023553
3: -0.0035027, -0.0020545, -0.0034625, -0.0019138, -0.0010720, 0.0008838
4: 0.0008601, 0.0014760, 0.0008003, 0.0014589, -0.0003758, 0.0004559
5: 0.0011185, 0.0051204, 0.0007299, 0.0050094, -0.0024422, 0.0029623
6: 0.0002412, 0.0012569, 0.0002694, 0.0013556, -0.0007519, 0.0006198
7: -0.0025135, 0.0001145, -0.0024406, 0.0003697, -0.0019453, 0.0016037
8: -0.0008860, 0.0004960, -0.0008477, 0.0006303, -0.0010230, 0.0008434
9: -0.0024390, -0.0008365, -0.0025947, -0.0008810, -0.0009780, 0.0011862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010036
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010036
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938022, 0.9961815, 0.9937444, 0.9963761, -0.0017194, 0.0015568
1: -0.0028083, -0.0022154, -0.0028227, -0.0021670, -0.0004284, 0.0003879
2: 0.0016866, 0.0048283, 0.0014297, 0.0049048, -0.0020558, 0.0022704
3: -0.0034708, -0.0020408, -0.0035056, -0.0019239, -0.0010334, 0.0009357
4: 0.0008543, 0.0014624, 0.0008046, 0.0014772, -0.0003979, 0.0004394
5: 0.0010807, 0.0050322, 0.0007576, 0.0051284, -0.0025856, 0.0028556
6: 0.0002636, 0.0012665, 0.0002392, 0.0013485, -0.0007248, 0.0006563
7: -0.0024556, 0.0001393, -0.0025188, 0.0003514, -0.0018752, 0.0016980
8: -0.0008555, 0.0005091, -0.0008887, 0.0006207, -0.0009862, 0.0008929
9: -0.0024542, -0.0008718, -0.0025836, -0.0008333, -0.0010354, 0.0011435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009928
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010021
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937605, 0.9961587, 0.9937444, 0.9963761, -0.0017506, 0.0015271
1: -0.0028187, -0.0022211, -0.0028227, -0.0021670, -0.0004362, 0.0003805
2: 0.0017166, 0.0048835, 0.0014297, 0.0049048, -0.0020165, 0.0023117
3: -0.0034959, -0.0020545, -0.0035056, -0.0019239, -0.0010522, 0.0009178
4: 0.0008601, 0.0014731, 0.0008046, 0.0014772, -0.0003903, 0.0004474
5: 0.0011185, 0.0051016, 0.0007576, 0.0051284, -0.0025362, 0.0029075
6: 0.0002460, 0.0012569, 0.0002392, 0.0013485, -0.0007380, 0.0006437
7: -0.0025011, 0.0001145, -0.0025188, 0.0003514, -0.0019093, 0.0016655
8: -0.0008795, 0.0004961, -0.0008887, 0.0006207, -0.0010041, 0.0008759
9: -0.0024391, -0.0008441, -0.0025836, -0.0008333, -0.0010156, 0.0011643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009946
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010037
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9938645, 0.9962469, 0.9938102, 0.9959704, -0.0013811, 0.0017250
1: -0.0027928, -0.0021991, -0.0028063, -0.0022680, -0.0003441, 0.0004298
2: 0.0016001, 0.0047462, 0.0019653, 0.0048179, -0.0022778, 0.0018237
3: -0.0034334, -0.0020014, -0.0034660, -0.0021676, -0.0008301, 0.0010368
4: 0.0008376, 0.0014465, 0.0009083, 0.0014604, -0.0004409, 0.0003530
5: 0.0009720, 0.0049289, 0.0014313, 0.0050191, -0.0028649, 0.0022938
6: 0.0002898, 0.0012941, 0.0002669, 0.0011776, -0.0005822, 0.0007271
7: -0.0023878, 0.0002107, -0.0024470, -0.0000909, -0.0015063, 0.0018814
8: -0.0008199, 0.0005467, -0.0008510, 0.0003880, -0.0007921, 0.0009894
9: -0.0024977, -0.0009132, -0.0023138, -0.0008771, -0.0011472, 0.0009185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009205, upper bound: 0.0009738
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009205, upper bound: 0.0009740
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9938465, 0.9962219, 0.9937921, 0.9959750, -0.0014050, 0.0017248
1: -0.0027972, -0.0022054, -0.0028108, -0.0022669, -0.0003501, 0.0004298
2: 0.0016333, 0.0047699, 0.0019593, 0.0048418, -0.0022776, 0.0018553
3: -0.0034442, -0.0020165, -0.0034769, -0.0021649, -0.0008444, 0.0010367
4: 0.0008440, 0.0014511, 0.0009071, 0.0014650, -0.0004408, 0.0003591
5: 0.0010137, 0.0049587, 0.0014237, 0.0050492, -0.0028646, 0.0023334
6: 0.0002823, 0.0012835, 0.0002593, 0.0011795, -0.0005923, 0.0007271
7: -0.0024073, 0.0001833, -0.0024667, -0.0000860, -0.0015323, 0.0018811
8: -0.0008301, 0.0005322, -0.0008614, 0.0003906, -0.0008058, 0.0009893
9: -0.0024810, -0.0009013, -0.0023168, -0.0008650, -0.0011471, 0.0009344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009300, upper bound: 0.0009907
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009300, upper bound: 0.0009909
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9938462, 0.9959931, -0.0014914, 0.0016567
1: -0.0028160, -0.0022102, -0.0027973, -0.0022624, -0.0003716, 0.0004128
2: 0.0016591, 0.0048696, 0.0019353, 0.0047704, -0.0021876, 0.0019694
3: -0.0034895, -0.0020283, -0.0034444, -0.0021540, -0.0008964, 0.0009957
4: 0.0008490, 0.0014704, 0.0009025, 0.0014512, -0.0004234, 0.0003812
5: 0.0010461, 0.0050841, 0.0013936, 0.0049593, -0.0027515, 0.0024770
6: 0.0002504, 0.0012753, 0.0002821, 0.0011871, -0.0006287, 0.0006984
7: -0.0024897, 0.0001620, -0.0024077, -0.0000662, -0.0016266, 0.0018069
8: -0.0008734, 0.0005210, -0.0008303, 0.0004010, -0.0008554, 0.0009502
9: -0.0024680, -0.0008511, -0.0023289, -0.0009010, -0.0011018, 0.0009919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009191, upper bound: 0.0009739
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009284, upper bound: 0.0009908
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9938033, 0.9959750, -0.0014627, 0.0016914
1: -0.0028160, -0.0022102, -0.0028080, -0.0022669, -0.0003645, 0.0004214
2: 0.0016591, 0.0048696, 0.0019593, 0.0048270, -0.0022334, 0.0019314
3: -0.0034895, -0.0020283, -0.0034702, -0.0021649, -0.0008791, 0.0010166
4: 0.0008490, 0.0014704, 0.0009071, 0.0014621, -0.0004323, 0.0003738
5: 0.0010461, 0.0050841, 0.0014237, 0.0050306, -0.0028091, 0.0024292
6: 0.0002504, 0.0012753, 0.0002640, 0.0011795, -0.0006166, 0.0007130
7: -0.0024897, 0.0001620, -0.0024545, -0.0000860, -0.0015952, 0.0018447
8: -0.0008734, 0.0005210, -0.0008550, 0.0003907, -0.0008389, 0.0009701
9: -0.0024680, -0.0008511, -0.0023168, -0.0008725, -0.0011249, 0.0009728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009191, upper bound: 0.0009760
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009284, upper bound: 0.0009924
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9938645, 0.9962469, 0.9937667, 0.9961540, -0.0015663, 0.0017621
1: -0.0027928, -0.0021991, -0.0028171, -0.0022223, -0.0003903, 0.0004391
2: 0.0016001, 0.0047462, 0.0017229, 0.0048754, -0.0023269, 0.0020683
3: -0.0034334, -0.0020014, -0.0034922, -0.0020573, -0.0009414, 0.0010591
4: 0.0008376, 0.0014465, 0.0008614, 0.0014715, -0.0004504, 0.0004003
5: 0.0009720, 0.0049289, 0.0011265, 0.0050914, -0.0029266, 0.0026014
6: 0.0002898, 0.0012941, 0.0002486, 0.0012549, -0.0006603, 0.0007428
7: -0.0023878, 0.0002107, -0.0024945, 0.0001093, -0.0017083, 0.0019219
8: -0.0008199, 0.0005467, -0.0008760, 0.0004933, -0.0008984, 0.0010107
9: -0.0024977, -0.0009132, -0.0024359, -0.0008481, -0.0011719, 0.0010417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009938, upper bound: 0.0009441
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009938, upper bound: 0.0009444
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9938465, 0.9962219, 0.9937491, 0.9961587, -0.0015957, 0.0017586
1: -0.0027972, -0.0022054, -0.0028215, -0.0022211, -0.0003976, 0.0004382
2: 0.0016333, 0.0047699, 0.0017166, 0.0048984, -0.0023222, 0.0021071
3: -0.0034442, -0.0020165, -0.0035027, -0.0020545, -0.0009591, 0.0010570
4: 0.0008440, 0.0014511, 0.0008601, 0.0014760, -0.0004495, 0.0004078
5: 0.0010137, 0.0049587, 0.0011185, 0.0051204, -0.0029208, 0.0026502
6: 0.0002823, 0.0012835, 0.0002412, 0.0012569, -0.0006726, 0.0007413
7: -0.0024073, 0.0001833, -0.0025135, 0.0001145, -0.0017403, 0.0019180
8: -0.0008301, 0.0005322, -0.0008860, 0.0004960, -0.0009152, 0.0010087
9: -0.0024810, -0.0009013, -0.0024390, -0.0008365, -0.0011696, 0.0010613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0009612
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0009613
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9938022, 0.9961815, -0.0016857, 0.0016977
1: -0.0028160, -0.0022102, -0.0028083, -0.0022154, -0.0004200, 0.0004230
2: 0.0016591, 0.0048696, 0.0016866, 0.0048283, -0.0022417, 0.0022259
3: -0.0034895, -0.0020283, -0.0034708, -0.0020408, -0.0010131, 0.0010203
4: 0.0008490, 0.0014704, 0.0008543, 0.0014624, -0.0004339, 0.0004308
5: 0.0010461, 0.0050841, 0.0010807, 0.0050322, -0.0028195, 0.0027996
6: 0.0002504, 0.0012753, 0.0002636, 0.0012665, -0.0007106, 0.0007156
7: -0.0024897, 0.0001620, -0.0024556, 0.0001393, -0.0018385, 0.0018515
8: -0.0008734, 0.0005210, -0.0008555, 0.0005091, -0.0009668, 0.0009737
9: -0.0024680, -0.0008511, -0.0024542, -0.0008718, -0.0011291, 0.0011211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0009442
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0009612
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937605, 0.9961587, -0.0016561, 0.0017257
1: -0.0028160, -0.0022102, -0.0028187, -0.0022211, -0.0004126, 0.0004300
2: 0.0016591, 0.0048696, 0.0017166, 0.0048835, -0.0022788, 0.0021868
3: -0.0034895, -0.0020283, -0.0034959, -0.0020545, -0.0009953, 0.0010372
4: 0.0008490, 0.0014704, 0.0008601, 0.0014731, -0.0004411, 0.0004233
5: 0.0010461, 0.0050841, 0.0011185, 0.0051016, -0.0028661, 0.0027504
6: 0.0002504, 0.0012753, 0.0002460, 0.0012569, -0.0006981, 0.0007275
7: -0.0024897, 0.0001620, -0.0025011, 0.0001145, -0.0018062, 0.0018821
8: -0.0008734, 0.0005210, -0.0008795, 0.0004961, -0.0009499, 0.0009898
9: -0.0024680, -0.0008511, -0.0024391, -0.0008441, -0.0011477, 0.0011014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0009459
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0009625
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9938354, 0.9964168, 0.9938102, 0.9959704, -0.0013936, 0.0018832
1: -0.0028000, -0.0021568, -0.0028063, -0.0022680, -0.0003472, 0.0004692
2: 0.0013760, 0.0047845, 0.0019653, 0.0048179, -0.0024868, 0.0018402
3: -0.0034508, -0.0018994, -0.0034660, -0.0021676, -0.0008376, 0.0011319
4: 0.0007942, 0.0014539, 0.0009083, 0.0014604, -0.0004813, 0.0003562
5: 0.0006900, 0.0049771, 0.0014313, 0.0050191, -0.0031277, 0.0023145
6: 0.0002776, 0.0013657, 0.0002669, 0.0011776, -0.0005874, 0.0007938
7: -0.0024194, 0.0003958, -0.0024470, -0.0000909, -0.0015199, 0.0020539
8: -0.0008365, 0.0006440, -0.0008510, 0.0003880, -0.0007993, 0.0010801
9: -0.0026106, -0.0008939, -0.0023138, -0.0008771, -0.0012525, 0.0009268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008900, upper bound: 0.0010369
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008900, upper bound: 0.0010370
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9938160, 0.9963927, 0.9937921, 0.9959750, -0.0014134, 0.0018831
1: -0.0028048, -0.0021628, -0.0028108, -0.0022669, -0.0003522, 0.0004692
2: 0.0014076, 0.0048102, 0.0019593, 0.0048418, -0.0024867, 0.0018664
3: -0.0034625, -0.0019138, -0.0034769, -0.0021649, -0.0008495, 0.0011318
4: 0.0008003, 0.0014589, 0.0009071, 0.0014650, -0.0004813, 0.0003612
5: 0.0007299, 0.0050094, 0.0014237, 0.0050492, -0.0031276, 0.0023474
6: 0.0002694, 0.0013556, 0.0002593, 0.0011795, -0.0005958, 0.0007938
7: -0.0024406, 0.0003697, -0.0024667, -0.0000860, -0.0015415, 0.0020538
8: -0.0008477, 0.0006303, -0.0008614, 0.0003906, -0.0008107, 0.0010801
9: -0.0025947, -0.0008810, -0.0023168, -0.0008650, -0.0012524, 0.0009400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008997, upper bound: 0.0010552
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008997, upper bound: 0.0010552
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9938462, 0.9959931, -0.0015026, 0.0018168
1: -0.0028227, -0.0021670, -0.0027973, -0.0022624, -0.0003744, 0.0004527
2: 0.0014297, 0.0049048, 0.0019353, 0.0047704, -0.0023991, 0.0019841
3: -0.0035056, -0.0019239, -0.0034444, -0.0021540, -0.0009031, 0.0010920
4: 0.0008046, 0.0014772, 0.0009025, 0.0014512, -0.0004643, 0.0003840
5: 0.0007576, 0.0051284, 0.0013936, 0.0049593, -0.0030175, 0.0024955
6: 0.0002392, 0.0013485, 0.0002821, 0.0011871, -0.0006334, 0.0007659
7: -0.0025188, 0.0003514, -0.0024077, -0.0000662, -0.0016388, 0.0019815
8: -0.0008887, 0.0006207, -0.0008303, 0.0004010, -0.0008618, 0.0010421
9: -0.0025836, -0.0008333, -0.0023289, -0.0009010, -0.0012083, 0.0009993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008877, upper bound: 0.0010369
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008968, upper bound: 0.0010552
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9938033, 0.9959750, -0.0014689, 0.0018504
1: -0.0028227, -0.0021670, -0.0028080, -0.0022669, -0.0003660, 0.0004611
2: 0.0014297, 0.0049048, 0.0019593, 0.0048270, -0.0024435, 0.0019397
3: -0.0035056, -0.0019239, -0.0034702, -0.0021649, -0.0008829, 0.0011122
4: 0.0008046, 0.0014772, 0.0009071, 0.0014621, -0.0004729, 0.0003754
5: 0.0007576, 0.0051284, 0.0014237, 0.0050306, -0.0030732, 0.0024397
6: 0.0002392, 0.0013485, 0.0002640, 0.0011795, -0.0006192, 0.0007800
7: -0.0025188, 0.0003514, -0.0024545, -0.0000860, -0.0016021, 0.0020181
8: -0.0008887, 0.0006207, -0.0008550, 0.0003907, -0.0008425, 0.0010613
9: -0.0025836, -0.0008333, -0.0023168, -0.0008725, -0.0012307, 0.0009769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008877, upper bound: 0.0010379
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008968, upper bound: 0.0010556
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9938354, 0.9964168, 0.9937667, 0.9961540, -0.0014478, 0.0017855
1: -0.0028000, -0.0021568, -0.0028171, -0.0022223, -0.0003608, 0.0004449
2: 0.0013760, 0.0047845, 0.0017229, 0.0048754, -0.0023577, 0.0019118
3: -0.0034508, -0.0018994, -0.0034922, -0.0020573, -0.0008702, 0.0010731
4: 0.0007942, 0.0014539, 0.0008614, 0.0014715, -0.0004563, 0.0003700
5: 0.0006900, 0.0049771, 0.0011265, 0.0050914, -0.0029654, 0.0024046
6: 0.0002776, 0.0013657, 0.0002486, 0.0012549, -0.0006103, 0.0007526
7: -0.0024194, 0.0003958, -0.0024945, 0.0001093, -0.0015791, 0.0019473
8: -0.0008365, 0.0006440, -0.0008760, 0.0004933, -0.0008304, 0.0010241
9: -0.0026106, -0.0008939, -0.0024359, -0.0008481, -0.0011875, 0.0009629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008971, upper bound: 0.0010385
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008971, upper bound: 0.0010385
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9938160, 0.9963927, 0.9937491, 0.9961587, -0.0014705, 0.0017836
1: -0.0028048, -0.0021628, -0.0028215, -0.0022211, -0.0003664, 0.0004444
2: 0.0014076, 0.0048102, 0.0017166, 0.0048984, -0.0023553, 0.0019417
3: -0.0034625, -0.0019138, -0.0035027, -0.0020545, -0.0008838, 0.0010720
4: 0.0008003, 0.0014589, 0.0008601, 0.0014760, -0.0004559, 0.0003758
5: 0.0007299, 0.0050094, 0.0011185, 0.0051204, -0.0029623, 0.0024422
6: 0.0002694, 0.0013556, 0.0002412, 0.0012569, -0.0006198, 0.0007519
7: -0.0024406, 0.0003697, -0.0025135, 0.0001145, -0.0016037, 0.0019453
8: -0.0008477, 0.0006303, -0.0008860, 0.0004960, -0.0008434, 0.0010230
9: -0.0025947, -0.0008810, -0.0024390, -0.0008365, -0.0011862, 0.0009780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009073, upper bound: 0.0010579
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009073, upper bound: 0.0010580
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9938022, 0.9961815, -0.0015568, 0.0017194
1: -0.0028227, -0.0021670, -0.0028083, -0.0022154, -0.0003879, 0.0004284
2: 0.0014297, 0.0049048, 0.0016866, 0.0048283, -0.0022704, 0.0020558
3: -0.0035056, -0.0019239, -0.0034708, -0.0020408, -0.0009357, 0.0010334
4: 0.0008046, 0.0014772, 0.0008543, 0.0014624, -0.0004394, 0.0003979
5: 0.0007576, 0.0051284, 0.0010807, 0.0050322, -0.0028556, 0.0025856
6: 0.0002392, 0.0013485, 0.0002636, 0.0012665, -0.0006563, 0.0007248
7: -0.0025188, 0.0003514, -0.0024556, 0.0001393, -0.0016980, 0.0018752
8: -0.0008887, 0.0006207, -0.0008555, 0.0005091, -0.0008929, 0.0009862
9: -0.0025836, -0.0008333, -0.0024542, -0.0008718, -0.0011435, 0.0010354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008943, upper bound: 0.0010384
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009040, upper bound: 0.0010580
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9937605, 0.9961587, -0.0015271, 0.0017506
1: -0.0028227, -0.0021670, -0.0028187, -0.0022211, -0.0003805, 0.0004362
2: 0.0014297, 0.0049048, 0.0017166, 0.0048835, -0.0023117, 0.0020165
3: -0.0035056, -0.0019239, -0.0034959, -0.0020545, -0.0009178, 0.0010522
4: 0.0008046, 0.0014772, 0.0008601, 0.0014731, -0.0004474, 0.0003903
5: 0.0007576, 0.0051284, 0.0011185, 0.0051016, -0.0029075, 0.0025362
6: 0.0002392, 0.0013485, 0.0002460, 0.0012569, -0.0006437, 0.0007380
7: -0.0025188, 0.0003514, -0.0025011, 0.0001145, -0.0016655, 0.0019093
8: -0.0008887, 0.0006207, -0.0008795, 0.0004961, -0.0008759, 0.0010041
9: -0.0025836, -0.0008333, -0.0024391, -0.0008441, -0.0011643, 0.0010156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008943, upper bound: 0.0010394
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009040, upper bound: 0.0010588
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938303, 0.9962212, 0.9938152, 0.9962225, -0.0013973, 0.0013983
1: -0.0028013, -0.0022055, -0.0028051, -0.0022052, -0.0003482, 0.0003484
2: 0.0016342, 0.0047912, 0.0016324, 0.0048114, -0.0018465, 0.0018452
3: -0.0034539, -0.0020169, -0.0034630, -0.0020161, -0.0008398, 0.0008404
4: 0.0008442, 0.0014552, 0.0008438, 0.0014591, -0.0003574, 0.0003571
5: 0.0010148, 0.0049855, 0.0010126, 0.0050109, -0.0023224, 0.0023207
6: 0.0002755, 0.0012833, 0.0002690, 0.0012838, -0.0005890, 0.0005895
7: -0.0024250, 0.0001825, -0.0024416, 0.0001840, -0.0015240, 0.0015251
8: -0.0008394, 0.0005319, -0.0008482, 0.0005326, -0.0008014, 0.0008020
9: -0.0024806, -0.0008905, -0.0024814, -0.0008804, -0.0009300, 0.0009293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009161, upper bound: 0.0009846
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009161, upper bound: 0.0009846
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938153, 0.9962254, 0.9937905, 0.9961991, -0.0014108, 0.0014480
1: -0.0028050, -0.0022045, -0.0028112, -0.0022111, -0.0003515, 0.0003608
2: 0.0016287, 0.0048112, 0.0016634, 0.0048439, -0.0019121, 0.0018630
3: -0.0034630, -0.0020144, -0.0034779, -0.0020302, -0.0008479, 0.0008703
4: 0.0008431, 0.0014591, 0.0008498, 0.0014654, -0.0003701, 0.0003606
5: 0.0010079, 0.0050106, 0.0010516, 0.0050518, -0.0024049, 0.0023431
6: 0.0002691, 0.0012850, 0.0002586, 0.0012739, -0.0005947, 0.0006104
7: -0.0024414, 0.0001871, -0.0024685, 0.0001584, -0.0015387, 0.0015792
8: -0.0008481, 0.0005342, -0.0008623, 0.0005192, -0.0008092, 0.0008305
9: -0.0024833, -0.0008805, -0.0024659, -0.0008640, -0.0009630, 0.0009383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009337, upper bound: 0.0009923
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009337, upper bound: 0.0009923
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9938119, 0.9962257, -0.0014815, 0.0014167
1: -0.0028160, -0.0022102, -0.0028059, -0.0022044, -0.0003691, 0.0003530
2: 0.0016591, 0.0048696, 0.0016282, 0.0048157, -0.0018708, 0.0019563
3: -0.0034895, -0.0020283, -0.0034650, -0.0020142, -0.0008904, 0.0008515
4: 0.0008490, 0.0014704, 0.0008430, 0.0014600, -0.0003621, 0.0003786
5: 0.0010461, 0.0050841, 0.0010073, 0.0050163, -0.0023529, 0.0024605
6: 0.0002504, 0.0012753, 0.0002676, 0.0012852, -0.0006245, 0.0005972
7: -0.0024897, 0.0001620, -0.0024452, 0.0001875, -0.0016158, 0.0015451
8: -0.0008734, 0.0005210, -0.0008500, 0.0005345, -0.0008497, 0.0008126
9: -0.0024680, -0.0008511, -0.0024836, -0.0008782, -0.0009422, 0.0009853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009237, upper bound: 0.0009751
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009319, upper bound: 0.0009923
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937711, 0.9962023, -0.0014475, 0.0014475
1: -0.0028160, -0.0022102, -0.0028160, -0.0022102, -0.0003607, 0.0003607
2: 0.0016591, 0.0048696, 0.0016591, 0.0048696, -0.0019115, 0.0019115
3: -0.0034895, -0.0020283, -0.0034895, -0.0020283, -0.0008700, 0.0008700
4: 0.0008490, 0.0014704, 0.0008490, 0.0014704, -0.0003700, 0.0003700
5: 0.0010461, 0.0050841, 0.0010461, 0.0050841, -0.0024041, 0.0024041
6: 0.0002504, 0.0012753, 0.0002504, 0.0012753, -0.0006102, 0.0006102
7: -0.0024897, 0.0001620, -0.0024897, 0.0001620, -0.0015787, 0.0015787
8: -0.0008734, 0.0005210, -0.0008734, 0.0005210, -0.0008302, 0.0008302
9: -0.0024680, -0.0008511, -0.0024680, -0.0008511, -0.0009627, 0.0009627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009237, upper bound: 0.0009769
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009319, upper bound: 0.0009923
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938303, 0.9962212, 0.9937888, 0.9963952, -0.0015887, 0.0014356
1: -0.0028013, -0.0022055, -0.0028116, -0.0021622, -0.0003959, 0.0003577
2: 0.0016342, 0.0047912, 0.0014044, 0.0048461, -0.0018957, 0.0020978
3: -0.0034539, -0.0020169, -0.0034789, -0.0019123, -0.0009548, 0.0008628
4: 0.0008442, 0.0014552, 0.0007997, 0.0014659, -0.0003669, 0.0004060
5: 0.0010148, 0.0049855, 0.0007258, 0.0050546, -0.0023843, 0.0026385
6: 0.0002755, 0.0012833, 0.0002579, 0.0013566, -0.0006697, 0.0006052
7: -0.0024250, 0.0001825, -0.0024703, 0.0003724, -0.0017327, 0.0015657
8: -0.0008394, 0.0005319, -0.0008633, 0.0006317, -0.0009112, 0.0008234
9: -0.0024806, -0.0008905, -0.0025963, -0.0008629, -0.0009548, 0.0010566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009846, upper bound: 0.0009556
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009846, upper bound: 0.0009556
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938153, 0.9962254, 0.9937624, 0.9963727, -0.0015955, 0.0014755
1: -0.0028050, -0.0022045, -0.0028182, -0.0021678, -0.0003976, 0.0003677
2: 0.0016287, 0.0048112, 0.0014342, 0.0048809, -0.0019484, 0.0021068
3: -0.0034630, -0.0020144, -0.0034947, -0.0019259, -0.0009589, 0.0008868
4: 0.0008431, 0.0014591, 0.0008055, 0.0014726, -0.0003771, 0.0004078
5: 0.0010079, 0.0050106, 0.0007632, 0.0050983, -0.0024506, 0.0026498
6: 0.0002691, 0.0012850, 0.0002468, 0.0013471, -0.0006726, 0.0006220
7: -0.0024414, 0.0001871, -0.0024990, 0.0003478, -0.0017401, 0.0016093
8: -0.0008481, 0.0005342, -0.0008784, 0.0006187, -0.0009151, 0.0008463
9: -0.0024833, -0.0008805, -0.0025813, -0.0008454, -0.0009813, 0.0010611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010037, upper bound: 0.0009618
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010037, upper bound: 0.0009619
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937840, 0.9963966, -0.0016743, 0.0014486
1: -0.0028160, -0.0022102, -0.0028128, -0.0021618, -0.0004172, 0.0003610
2: 0.0016591, 0.0048696, 0.0014025, 0.0048524, -0.0019129, 0.0022109
3: -0.0034895, -0.0020283, -0.0034817, -0.0019115, -0.0010063, 0.0008707
4: 0.0008490, 0.0014704, 0.0007993, 0.0014671, -0.0003702, 0.0004279
5: 0.0010461, 0.0050841, 0.0007234, 0.0050625, -0.0024059, 0.0027807
6: 0.0002504, 0.0012753, 0.0002559, 0.0013572, -0.0007058, 0.0006106
7: -0.0024897, 0.0001620, -0.0024755, 0.0003740, -0.0018260, 0.0015799
8: -0.0008734, 0.0005210, -0.0008660, 0.0006325, -0.0009603, 0.0008309
9: -0.0024680, -0.0008511, -0.0025973, -0.0008597, -0.0009634, 0.0011135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009940, upper bound: 0.0009452
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010022, upper bound: 0.0009618
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937711, 0.9962023, 0.9937444, 0.9963761, -0.0016395, 0.0014742
1: -0.0028160, -0.0022102, -0.0028227, -0.0021670, -0.0004085, 0.0003673
2: 0.0016591, 0.0048696, 0.0014297, 0.0049048, -0.0019467, 0.0021649
3: -0.0034895, -0.0020283, -0.0035056, -0.0019239, -0.0009854, 0.0008861
4: 0.0008490, 0.0014704, 0.0008046, 0.0014772, -0.0003768, 0.0004190
5: 0.0010461, 0.0050841, 0.0007576, 0.0051284, -0.0024484, 0.0027229
6: 0.0002504, 0.0012753, 0.0002392, 0.0013485, -0.0006911, 0.0006214
7: -0.0024897, 0.0001620, -0.0025188, 0.0003514, -0.0017881, 0.0016079
8: -0.0008734, 0.0005210, -0.0008887, 0.0006207, -0.0009403, 0.0008456
9: -0.0024680, -0.0008511, -0.0025836, -0.0008333, -0.0009805, 0.0010904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009940, upper bound: 0.0009463
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010022, upper bound: 0.0009628
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9937888, 0.9963952, 0.9938303, 0.9962212, -0.0014356, 0.0015887
1: -0.0028116, -0.0021622, -0.0028013, -0.0022055, -0.0003577, 0.0003959
2: 0.0014044, 0.0048461, 0.0016342, 0.0047912, -0.0020978, 0.0018957
3: -0.0034789, -0.0019123, -0.0034539, -0.0020169, -0.0008628, 0.0009548
4: 0.0007997, 0.0014659, 0.0008442, 0.0014552, -0.0004060, 0.0003669
5: 0.0007258, 0.0050546, 0.0010148, 0.0049855, -0.0026385, 0.0023843
6: 0.0002579, 0.0013566, 0.0002755, 0.0012833, -0.0006052, 0.0006697
7: -0.0024703, 0.0003724, -0.0024250, 0.0001825, -0.0015657, 0.0017327
8: -0.0008633, 0.0006317, -0.0008394, 0.0005319, -0.0008234, 0.0009112
9: -0.0025963, -0.0008629, -0.0024806, -0.0008905, -0.0010566, 0.0009548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008908, upper bound: 0.0010384
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008908, upper bound: 0.0010384
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9937624, 0.9963727, 0.9938153, 0.9962254, -0.0014755, 0.0015955
1: -0.0028182, -0.0021678, -0.0028050, -0.0022045, -0.0003677, 0.0003976
2: 0.0014342, 0.0048809, 0.0016287, 0.0048112, -0.0021068, 0.0019484
3: -0.0034947, -0.0019259, -0.0034630, -0.0020144, -0.0008868, 0.0009589
4: 0.0008055, 0.0014726, 0.0008431, 0.0014591, -0.0004078, 0.0003771
5: 0.0007632, 0.0050983, 0.0010079, 0.0050106, -0.0026498, 0.0024506
6: 0.0002468, 0.0013471, 0.0002691, 0.0012850, -0.0006220, 0.0006726
7: -0.0024990, 0.0003478, -0.0024414, 0.0001871, -0.0016093, 0.0017401
8: -0.0008784, 0.0006187, -0.0008481, 0.0005342, -0.0008463, 0.0009151
9: -0.0025813, -0.0008454, -0.0024833, -0.0008805, -0.0010611, 0.0009813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010586
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010586
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9937840, 0.9963966, 0.9937711, 0.9962023, -0.0014486, 0.0016743
1: -0.0028128, -0.0021618, -0.0028160, -0.0022102, -0.0003610, 0.0004172
2: 0.0014025, 0.0048524, 0.0016591, 0.0048696, -0.0022109, 0.0019129
3: -0.0034817, -0.0019115, -0.0034895, -0.0020283, -0.0008707, 0.0010063
4: 0.0007993, 0.0014671, 0.0008490, 0.0014704, -0.0004279, 0.0003702
5: 0.0007234, 0.0050625, 0.0010461, 0.0050841, -0.0027807, 0.0024059
6: 0.0002559, 0.0013572, 0.0002504, 0.0012753, -0.0006106, 0.0007058
7: -0.0024755, 0.0003740, -0.0024897, 0.0001620, -0.0015799, 0.0018260
8: -0.0008660, 0.0006325, -0.0008734, 0.0005210, -0.0008309, 0.0009603
9: -0.0025973, -0.0008597, -0.0024680, -0.0008511, -0.0011135, 0.0009634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008863, upper bound: 0.0010480
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010558
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9937711, 0.9962023, -0.0014742, 0.0016395
1: -0.0028227, -0.0021670, -0.0028160, -0.0022102, -0.0003673, 0.0004085
2: 0.0014297, 0.0049048, 0.0016591, 0.0048696, -0.0021649, 0.0019467
3: -0.0035056, -0.0019239, -0.0034895, -0.0020283, -0.0008861, 0.0009854
4: 0.0008046, 0.0014772, 0.0008490, 0.0014704, -0.0004190, 0.0003768
5: 0.0007576, 0.0051284, 0.0010461, 0.0050841, -0.0027229, 0.0024484
6: 0.0002392, 0.0013485, 0.0002504, 0.0012753, -0.0006214, 0.0006911
7: -0.0025188, 0.0003514, -0.0024897, 0.0001620, -0.0016079, 0.0017881
8: -0.0008887, 0.0006207, -0.0008734, 0.0005210, -0.0008456, 0.0009403
9: -0.0025836, -0.0008333, -0.0024680, -0.0008511, -0.0010904, 0.0009805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008863, upper bound: 0.0010491
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010564
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938017, 0.9963920, 0.9937888, 0.9963952, -0.0014539, 0.0014544
1: -0.0028084, -0.0021630, -0.0028116, -0.0021622, -0.0003623, 0.0003624
2: 0.0014087, 0.0048291, 0.0014044, 0.0048461, -0.0019205, 0.0019198
3: -0.0034711, -0.0019143, -0.0034789, -0.0019123, -0.0008738, 0.0008741
4: 0.0008005, 0.0014626, 0.0007997, 0.0014659, -0.0003717, 0.0003716
5: 0.0007312, 0.0050332, 0.0007258, 0.0050546, -0.0024155, 0.0024147
6: 0.0002633, 0.0013552, 0.0002579, 0.0013566, -0.0006129, 0.0006131
7: -0.0024563, 0.0003688, -0.0024703, 0.0003724, -0.0015857, 0.0015863
8: -0.0008559, 0.0006298, -0.0008633, 0.0006317, -0.0008339, 0.0008342
9: -0.0025941, -0.0008714, -0.0025963, -0.0008629, -0.0009673, 0.0009669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008949, upper bound: 0.0010493
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008949, upper bound: 0.0010493
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9937871, 0.9963964, 0.9937624, 0.9963727, -0.0014649, 0.0015008
1: -0.0028120, -0.0021619, -0.0028182, -0.0021678, -0.0003650, 0.0003740
2: 0.0014029, 0.0048483, 0.0014342, 0.0048809, -0.0019818, 0.0019344
3: -0.0034799, -0.0019117, -0.0034947, -0.0019259, -0.0008804, 0.0009020
4: 0.0007994, 0.0014663, 0.0008055, 0.0014726, -0.0003836, 0.0003744
5: 0.0007240, 0.0050574, 0.0007632, 0.0050983, -0.0024926, 0.0024329
6: 0.0002572, 0.0013571, 0.0002468, 0.0013471, -0.0006175, 0.0006326
7: -0.0024721, 0.0003735, -0.0024990, 0.0003478, -0.0015977, 0.0016368
8: -0.0008642, 0.0006323, -0.0008784, 0.0006187, -0.0008402, 0.0008608
9: -0.0025970, -0.0008617, -0.0025813, -0.0008454, -0.0009981, 0.0009742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009101, upper bound: 0.0010588
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009101, upper bound: 0.0010588
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9937840, 0.9963966, -0.0015369, 0.0014705
1: -0.0028227, -0.0021670, -0.0028128, -0.0021618, -0.0003830, 0.0003664
2: 0.0014297, 0.0049048, 0.0014025, 0.0048524, -0.0019418, 0.0020295
3: -0.0035056, -0.0019239, -0.0034817, -0.0019115, -0.0009237, 0.0008838
4: 0.0008046, 0.0014772, 0.0007993, 0.0014671, -0.0003758, 0.0003928
5: 0.0007576, 0.0051284, 0.0007234, 0.0050625, -0.0024423, 0.0025526
6: 0.0002392, 0.0013485, 0.0002559, 0.0013572, -0.0006479, 0.0006199
7: -0.0025188, 0.0003514, -0.0024755, 0.0003740, -0.0016762, 0.0016038
8: -0.0008887, 0.0006207, -0.0008660, 0.0006325, -0.0008815, 0.0008434
9: -0.0025836, -0.0008333, -0.0025973, -0.0008597, -0.0009780, 0.0010222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008963, upper bound: 0.0010387
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009069, upper bound: 0.0010588
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937444, 0.9963761, 0.9937444, 0.9963761, -0.0015014, 0.0015014
1: -0.0028227, -0.0021670, -0.0028227, -0.0021670, -0.0003741, 0.0003741
2: 0.0014297, 0.0049048, 0.0014297, 0.0049048, -0.0019826, 0.0019826
3: -0.0035056, -0.0019239, -0.0035056, -0.0019239, -0.0009024, 0.0009024
4: 0.0008046, 0.0014772, 0.0008046, 0.0014772, -0.0003837, 0.0003837
5: 0.0007576, 0.0051284, 0.0007576, 0.0051284, -0.0024936, 0.0024936
6: 0.0002392, 0.0013485, 0.0002392, 0.0013485, -0.0006329, 0.0006329
7: -0.0025188, 0.0003514, -0.0025188, 0.0003514, -0.0016375, 0.0016375
8: -0.0008887, 0.0006207, -0.0008887, 0.0006207, -0.0008612, 0.0008612
9: -0.0025836, -0.0008333, -0.0025836, -0.0008333, -0.0009986, 0.0009986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008963, upper bound: 0.0010396
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009069, upper bound: 0.0010599
time: 0.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009191
IS_A1_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009194
IS_A1_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0009287
IS_A1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0009288
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009195, upper bound: 0.0009113
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009287, upper bound: 0.0009288
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009195, upper bound: 0.0009159
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009287, upper bound: 0.0009315
IS_A1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009831, upper bound: 0.0008892
IS_A1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009831, upper bound: 0.0008893
IS_A1_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0008982
IS_A1_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0008982
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0008838
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0008983
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0008862
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0008999
IS_A1_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009831
IS_A1_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009831
IS_A1_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010018
IS_A1_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010018
IS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008837, upper bound: 0.0009915
IS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010002
IS_A1_B1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008837, upper bound: 0.0009935
IS_A1_B1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008982, upper bound: 0.0010013
IS_A1_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008979, upper bound: 0.0009839
IS_A1_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008979, upper bound: 0.0009840
IS_A1_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0010021
IS_A1_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009075, upper bound: 0.0010021
IS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008902, upper bound: 0.0009929
IS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009052, upper bound: 0.0010021
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008959, upper bound: 0.0009854
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009052, upper bound: 0.0010037
IS_A1_B2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009206
IS_A1_B2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009206
IS_A1_B2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009300
IS_A1_B2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009300
IS_A1_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009191
IS_A1_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009285
IS_A1_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009231
IS_A1_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009907, upper bound: 0.0009310
IS_A1_B2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009938
IS_A1_B2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009938
IS_A1_B2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010018
IS_A1_B2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010018
IS_A1_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009915
IS_A1_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010002
IS_A1_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009442, upper bound: 0.0009935
IS_A1_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009612, upper bound: 0.0010013
IS_A1_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008900
IS_A1_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008900
IS_A1_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008997
IS_A1_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008997
IS_A1_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008877
IS_A1_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008967
IS_A1_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010369, upper bound: 0.0008906
IS_A1_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010552, upper bound: 0.0008988
IS_A1_B2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009950
IS_A1_B2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009950
IS_A1_B2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010036
IS_A1_B2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010036
IS_A1_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009928
IS_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010021
IS_A1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009501, upper bound: 0.0009946
IS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009679, upper bound: 0.0010037
IS_A2_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009205, upper bound: 0.0009738
IS_A2_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009205, upper bound: 0.0009740
IS_A2_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009300, upper bound: 0.0009907
IS_A2_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009300, upper bound: 0.0009909
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009191, upper bound: 0.0009739
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009284, upper bound: 0.0009908
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009191, upper bound: 0.0009760
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009284, upper bound: 0.0009924
IS_A2_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009938, upper bound: 0.0009441
IS_A2_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009938, upper bound: 0.0009444
IS_A2_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0009612
IS_A2_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010018, upper bound: 0.0009613
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0009442
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0009612
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009915, upper bound: 0.0009459
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010002, upper bound: 0.0009625
IS_A2_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008900, upper bound: 0.0010369
IS_A2_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008900, upper bound: 0.0010370
IS_A2_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008997, upper bound: 0.0010552
IS_A2_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008997, upper bound: 0.0010552
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008877, upper bound: 0.0010369
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008968, upper bound: 0.0010552
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008877, upper bound: 0.0010379
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008968, upper bound: 0.0010556
IS_A2_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008971, upper bound: 0.0010385
IS_A2_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008971, upper bound: 0.0010385
IS_A2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009073, upper bound: 0.0010579
IS_A2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009073, upper bound: 0.0010580
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008943, upper bound: 0.0010384
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009040, upper bound: 0.0010580
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008943, upper bound: 0.0010394
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009040, upper bound: 0.0010588
IS_A2_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009161, upper bound: 0.0009846
IS_A2_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009161, upper bound: 0.0009846
IS_A2_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009337, upper bound: 0.0009923
IS_A2_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009337, upper bound: 0.0009923
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009237, upper bound: 0.0009751
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009319, upper bound: 0.0009923
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009237, upper bound: 0.0009769
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009319, upper bound: 0.0009923
IS_A2_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009846, upper bound: 0.0009556
IS_A2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009846, upper bound: 0.0009556
IS_A2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010037, upper bound: 0.0009618
IS_A2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010037, upper bound: 0.0009619
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009940, upper bound: 0.0009452
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010022, upper bound: 0.0009618
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009940, upper bound: 0.0009463
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0010022, upper bound: 0.0009628
IS_A2_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008908, upper bound: 0.0010384
IS_A2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008908, upper bound: 0.0010384
IS_A2_B2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010586
IS_A2_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010586
IS_A2_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008863, upper bound: 0.0010480
IS_A2_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010558
IS_A2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008863, upper bound: 0.0010491
IS_A2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008991, upper bound: 0.0010564
IS_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008949, upper bound: 0.0010493
IS_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008949, upper bound: 0.0010493
IS_A2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009101, upper bound: 0.0010588
IS_A2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009101, upper bound: 0.0010588
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008963, upper bound: 0.0010387
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009069, upper bound: 0.0010588
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0008963, upper bound: 0.0010396
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 0, lower bound: -0.0009069, upper bound: 0.0010599

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9938651, 0.9959879, 0.9939033, 0.9960083, -0.0012862, 0.0012305
1: -0.0027926, -0.0022637, -0.0027831, -0.0022586, -0.0003205, 0.0003066
2: 0.0019422, 0.0047453, 0.0019154, 0.0046949, -0.0016249, 0.0016984
3: -0.0034330, -0.0021571, -0.0034100, -0.0021449, -0.0007731, 0.0007396
4: 0.0009038, 0.0014463, 0.0008986, 0.0014366, -0.0003145, 0.0003287
5: 0.0014022, 0.0049278, 0.0013685, 0.0048644, -0.0020437, 0.0021362
6: 0.0002901, 0.0011849, 0.0003062, 0.0011935, -0.0005422, 0.0005187
7: -0.0023871, -0.0000718, -0.0023454, -0.0000497, -0.0014028, 0.0013421
8: -0.0008195, 0.0003981, -0.0007976, 0.0004097, -0.0007377, 0.0007058
9: -0.0023254, -0.0009136, -0.0023389, -0.0009390, -0.0008184, 0.0008554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005837, upper bound: 0.0006298
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009183
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009191
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9938651, 0.9959879, 0.9938649, 0.9959903, -0.0012843, 0.0012804
1: -0.0027926, -0.0022637, -0.0027927, -0.0022631, -0.0003200, 0.0003190
2: 0.0019422, 0.0047453, 0.0019391, 0.0047456, -0.0016907, 0.0016959
3: -0.0034330, -0.0021571, -0.0034331, -0.0021557, -0.0007719, 0.0007695
4: 0.0009038, 0.0014463, 0.0009032, 0.0014464, -0.0003272, 0.0003282
5: 0.0014022, 0.0049278, 0.0013984, 0.0049282, -0.0021264, 0.0021330
6: 0.0002901, 0.0011849, 0.0002900, 0.0011859, -0.0005414, 0.0005397
7: -0.0023871, -0.0000718, -0.0023873, -0.0000693, -0.0014007, 0.0013964
8: -0.0008195, 0.0003981, -0.0008196, 0.0003994, -0.0007366, 0.0007344
9: -0.0023254, -0.0009136, -0.0023270, -0.0009135, -0.0008515, 0.0008542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005837, upper bound: 0.0006299
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009187
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009124, upper bound: 0.0009194
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9938492, 0.9959928, 0.9938784, 0.9959891, -0.0012947, 0.0012675
1: -0.0027966, -0.0022625, -0.0027893, -0.0022634, -0.0003226, 0.0003158
2: 0.0019359, 0.0047663, 0.0019407, 0.0047279, -0.0016738, 0.0017096
3: -0.0034426, -0.0021542, -0.0034250, -0.0021565, -0.0007781, 0.0007618
4: 0.0009026, 0.0014504, 0.0009035, 0.0014430, -0.0003240, 0.0003309
5: 0.0013942, 0.0049543, 0.0014004, 0.0049059, -0.0021052, 0.0021502
6: 0.0002834, 0.0011870, 0.0002957, 0.0011854, -0.0005457, 0.0005343
7: -0.0024044, -0.0000666, -0.0023726, -0.0000706, -0.0014120, 0.0013824
8: -0.0008286, 0.0004008, -0.0008119, 0.0003987, -0.0007426, 0.0007270
9: -0.0023286, -0.0009030, -0.0023262, -0.0009224, -0.0008430, 0.0008610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007050, upper bound: 0.0007349
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006226, upper bound: 0.0006226
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9938492, 0.9959928, 0.9938385, 0.9959711, -0.0012955, 0.0013247
1: -0.0027966, -0.0022625, -0.0027992, -0.0022678, -0.0003228, 0.0003301
2: 0.0019359, 0.0047663, 0.0019644, 0.0047804, -0.0017493, 0.0017107
3: -0.0034426, -0.0021542, -0.0034490, -0.0021672, -0.0007786, 0.0007962
4: 0.0009026, 0.0014504, 0.0009081, 0.0014531, -0.0003386, 0.0003311
5: 0.0013942, 0.0049543, 0.0014301, 0.0049720, -0.0022002, 0.0021516
6: 0.0002834, 0.0011870, 0.0002789, 0.0011778, -0.0005461, 0.0005584
7: -0.0024044, -0.0000666, -0.0024161, -0.0000902, -0.0014129, 0.0014448
8: -0.0008286, 0.0004008, -0.0008347, 0.0003884, -0.0007430, 0.0007598
9: -0.0023286, -0.0009030, -0.0023143, -0.0008959, -0.0008810, 0.0008616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007349, upper bound: 0.0007423
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006226, upper bound: 0.0006589
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938649, 0.9959903, 0.9938651, 0.9959879, -0.0012804, 0.0012843
1: -0.0027927, -0.0022631, -0.0027926, -0.0022637, -0.0003190, 0.0003200
2: 0.0019391, 0.0047456, 0.0019422, 0.0047453, -0.0016959, 0.0016907
3: -0.0034331, -0.0021557, -0.0034330, -0.0021571, -0.0007695, 0.0007719
4: 0.0009032, 0.0014464, 0.0009038, 0.0014463, -0.0003282, 0.0003272
5: 0.0013984, 0.0049282, 0.0014022, 0.0049278, -0.0021330, 0.0021264
6: 0.0002900, 0.0011859, 0.0002901, 0.0011849, -0.0005397, 0.0005414
7: -0.0023873, -0.0000693, -0.0023871, -0.0000718, -0.0013964, 0.0014007
8: -0.0008196, 0.0003994, -0.0008195, 0.0003981, -0.0007344, 0.0007366
9: -0.0023270, -0.0009135, -0.0023254, -0.0009136, -0.0008542, 0.0008515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005492, upper bound: 0.0005758
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006299, upper bound: 0.0005843
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009120, upper bound: 0.0009124
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009120, upper bound: 0.0009124
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938385, 0.9959711, 0.9938492, 0.9959928, -0.0013247, 0.0012955
1: -0.0027992, -0.0022678, -0.0027966, -0.0022625, -0.0003301, 0.0003228
2: 0.0019644, 0.0047804, 0.0019359, 0.0047663, -0.0017107, 0.0017493
3: -0.0034490, -0.0021672, -0.0034426, -0.0021542, -0.0007962, 0.0007786
4: 0.0009081, 0.0014531, 0.0009026, 0.0014504, -0.0003311, 0.0003386
5: 0.0014301, 0.0049720, 0.0013942, 0.0049543, -0.0021516, 0.0022002
6: 0.0002789, 0.0011778, 0.0002834, 0.0011870, -0.0005584, 0.0005461
7: -0.0024161, -0.0000902, -0.0024044, -0.0000666, -0.0014448, 0.0014129
8: -0.0008347, 0.0003884, -0.0008286, 0.0004008, -0.0007598, 0.0007430
9: -0.0023143, -0.0008959, -0.0023286, -0.0009030, -0.0008616, 0.0008810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007423, upper bound: 0.0007487
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006589, upper bound: 0.0006403
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938649, 0.9959903, 0.9938247, 0.9959700, -0.0012472, 0.0013090
1: -0.0027927, -0.0022631, -0.0028027, -0.0022681, -0.0003108, 0.0003262
2: 0.0019391, 0.0047456, 0.0019658, 0.0047987, -0.0017286, 0.0016469
3: -0.0034331, -0.0021557, -0.0034573, -0.0021679, -0.0007496, 0.0007868
4: 0.0009032, 0.0014464, 0.0009084, 0.0014567, -0.0003346, 0.0003188
5: 0.0013984, 0.0049282, 0.0014319, 0.0049950, -0.0021741, 0.0020714
6: 0.0002900, 0.0011859, 0.0002731, 0.0011774, -0.0005257, 0.0005518
7: -0.0023873, -0.0000693, -0.0024312, -0.0000914, -0.0013603, 0.0014277
8: -0.0008196, 0.0003994, -0.0008427, 0.0003878, -0.0007154, 0.0007508
9: -0.0023270, -0.0009135, -0.0023135, -0.0008867, -0.0008706, 0.0008295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007240, upper bound: 0.0006946
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005623, upper bound: 0.0005584
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938385, 0.9959711, 0.9938066, 0.9959747, -0.0012912, 0.0013226
1: -0.0027992, -0.0022678, -0.0028072, -0.0022670, -0.0003217, 0.0003296
2: 0.0019644, 0.0047804, 0.0019598, 0.0048226, -0.0017465, 0.0017050
3: -0.0034490, -0.0021672, -0.0034682, -0.0021651, -0.0007760, 0.0007949
4: 0.0009081, 0.0014531, 0.0009072, 0.0014613, -0.0003380, 0.0003300
5: 0.0014301, 0.0049720, 0.0014244, 0.0050250, -0.0021966, 0.0021444
6: 0.0002789, 0.0011778, 0.0002654, 0.0011793, -0.0005443, 0.0005575
7: -0.0024161, -0.0000902, -0.0024509, -0.0000864, -0.0014082, 0.0014425
8: -0.0008347, 0.0003884, -0.0008530, 0.0003904, -0.0007406, 0.0007586
9: -0.0023143, -0.0008959, -0.0023166, -0.0008747, -0.0008796, 0.0008587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008014, upper bound: 0.0007910
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007534, upper bound: 0.0007532
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9938651, 0.9959879, 0.9938560, 0.9961986, -0.0014857, 0.0012756
1: -0.0027926, -0.0022637, -0.0027949, -0.0022112, -0.0003702, 0.0003178
2: 0.0019422, 0.0047453, 0.0016640, 0.0047573, -0.0016844, 0.0019618
3: -0.0034330, -0.0021571, -0.0034385, -0.0020305, -0.0008929, 0.0007667
4: 0.0009038, 0.0014463, 0.0008500, 0.0014487, -0.0003260, 0.0003797
5: 0.0014022, 0.0049278, 0.0010523, 0.0049429, -0.0021185, 0.0024675
6: 0.0002901, 0.0011849, 0.0002863, 0.0012737, -0.0006263, 0.0005377
7: -0.0023871, -0.0000718, -0.0023970, 0.0001579, -0.0016204, 0.0013912
8: -0.0008195, 0.0003981, -0.0008247, 0.0005189, -0.0008521, 0.0007316
9: -0.0023254, -0.0009136, -0.0024656, -0.0009076, -0.0008483, 0.0009881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006353, upper bound: 0.0006253
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009832, upper bound: 0.0008891
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009832, upper bound: 0.0008892
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9938651, 0.9959879, 0.9938185, 0.9961754, -0.0014776, 0.0013252
1: -0.0027926, -0.0022637, -0.0028043, -0.0022169, -0.0003682, 0.0003302
2: 0.0019422, 0.0047453, 0.0016946, 0.0048071, -0.0017499, 0.0019512
3: -0.0034330, -0.0021571, -0.0034611, -0.0020444, -0.0008881, 0.0007965
4: 0.0009038, 0.0014463, 0.0008559, 0.0014583, -0.0003387, 0.0003777
5: 0.0014022, 0.0049278, 0.0010908, 0.0050055, -0.0022009, 0.0024541
6: 0.0002901, 0.0011849, 0.0002704, 0.0012640, -0.0006229, 0.0005586
7: -0.0023871, -0.0000718, -0.0024380, 0.0001327, -0.0016116, 0.0014453
8: -0.0008195, 0.0003981, -0.0008463, 0.0005056, -0.0008475, 0.0007601
9: -0.0023254, -0.0009136, -0.0024501, -0.0008825, -0.0008813, 0.0009827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006353, upper bound: 0.0006275
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009832, upper bound: 0.0008893
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009832, upper bound: 0.0008893
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9938492, 0.9959928, 0.9938353, 0.9961774, -0.0014886, 0.0013104
1: -0.0027966, -0.0022625, -0.0028001, -0.0022165, -0.0003709, 0.0003265
2: 0.0019359, 0.0047663, 0.0016921, 0.0047848, -0.0017303, 0.0019656
3: -0.0034426, -0.0021542, -0.0034510, -0.0020433, -0.0008947, 0.0007876
4: 0.0009026, 0.0014504, 0.0008554, 0.0014540, -0.0003349, 0.0003804
5: 0.0013942, 0.0049543, 0.0010877, 0.0049775, -0.0021763, 0.0024723
6: 0.0002834, 0.0011870, 0.0002775, 0.0012648, -0.0006275, 0.0005524
7: -0.0024044, -0.0000666, -0.0024197, 0.0001347, -0.0016235, 0.0014291
8: -0.0008286, 0.0004008, -0.0008366, 0.0005067, -0.0008538, 0.0007516
9: -0.0023286, -0.0009030, -0.0024514, -0.0008937, -0.0008715, 0.0009900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008078, upper bound: 0.0007298
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007292, upper bound: 0.0006236
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9938492, 0.9959928, 0.9937955, 0.9961548, -0.0014809, 0.0013628
1: -0.0027966, -0.0022625, -0.0028100, -0.0022221, -0.0003690, 0.0003396
2: 0.0019359, 0.0047663, 0.0017219, 0.0048374, -0.0017996, 0.0019555
3: -0.0034426, -0.0021542, -0.0034749, -0.0020569, -0.0008901, 0.0008191
4: 0.0009026, 0.0014504, 0.0008612, 0.0014642, -0.0003483, 0.0003785
5: 0.0013942, 0.0049543, 0.0011251, 0.0050436, -0.0022634, 0.0024595
6: 0.0002834, 0.0011870, 0.0002607, 0.0012553, -0.0006242, 0.0005745
7: -0.0024044, -0.0000666, -0.0024631, 0.0001101, -0.0016151, 0.0014863
8: -0.0008286, 0.0004008, -0.0008595, 0.0004938, -0.0008494, 0.0007816
9: -0.0023286, -0.0009030, -0.0024364, -0.0008673, -0.0009064, 0.0009849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008574, upper bound: 0.0007422
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007292, upper bound: 0.0006589
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938649, 0.9959903, 0.9938200, 0.9961761, -0.0014740, 0.0013282
1: -0.0027927, -0.0022631, -0.0028038, -0.0022168, -0.0003673, 0.0003309
2: 0.0019391, 0.0047456, 0.0016937, 0.0048049, -0.0017538, 0.0019464
3: -0.0034331, -0.0021557, -0.0034601, -0.0020440, -0.0008859, 0.0007983
4: 0.0009032, 0.0014464, 0.0008557, 0.0014579, -0.0003394, 0.0003767
5: 0.0013984, 0.0049282, 0.0010897, 0.0050027, -0.0022058, 0.0024481
6: 0.0002900, 0.0011859, 0.0002711, 0.0012643, -0.0006213, 0.0005599
7: -0.0023873, -0.0000693, -0.0024362, 0.0001334, -0.0016076, 0.0014485
8: -0.0008196, 0.0003994, -0.0008453, 0.0005060, -0.0008454, 0.0007618
9: -0.0023270, -0.0009135, -0.0024506, -0.0008836, -0.0008833, 0.0009803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006258, upper bound: 0.0005758
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007486, upper bound: 0.0005869
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009823, upper bound: 0.0008860
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009823, upper bound: 0.0008859
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938385, 0.9959711, 0.9938053, 0.9961811, -0.0015207, 0.0013361
1: -0.0027992, -0.0022678, -0.0028075, -0.0022155, -0.0003789, 0.0003329
2: 0.0019644, 0.0047804, 0.0016871, 0.0048242, -0.0017644, 0.0020080
3: -0.0034490, -0.0021672, -0.0034689, -0.0020410, -0.0009140, 0.0008031
4: 0.0009081, 0.0014531, 0.0008544, 0.0014616, -0.0003415, 0.0003887
5: 0.0014301, 0.0049720, 0.0010814, 0.0050270, -0.0022191, 0.0025256
6: 0.0002789, 0.0011778, 0.0002649, 0.0012664, -0.0006410, 0.0005632
7: -0.0024161, -0.0000902, -0.0024522, 0.0001389, -0.0016585, 0.0014573
8: -0.0008347, 0.0003884, -0.0008537, 0.0005089, -0.0008722, 0.0007664
9: -0.0023143, -0.0008959, -0.0024539, -0.0008739, -0.0008886, 0.0010114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008336, upper bound: 0.0007428
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007539, upper bound: 0.0006407
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938649, 0.9959903, 0.9937815, 0.9961535, -0.0014399, 0.0013473
1: -0.0027927, -0.0022631, -0.0028134, -0.0022224, -0.0003588, 0.0003357
2: 0.0019391, 0.0047456, 0.0017235, 0.0048558, -0.0017790, 0.0019014
3: -0.0034331, -0.0021557, -0.0034833, -0.0020576, -0.0008654, 0.0008097
4: 0.0009032, 0.0014464, 0.0008615, 0.0014677, -0.0003443, 0.0003680
5: 0.0013984, 0.0049282, 0.0011272, 0.0050667, -0.0022376, 0.0023915
6: 0.0002900, 0.0011859, 0.0002548, 0.0012547, -0.0006070, 0.0005679
7: -0.0023873, -0.0000693, -0.0024783, 0.0001088, -0.0015705, 0.0014694
8: -0.0008196, 0.0003994, -0.0008674, 0.0004931, -0.0008259, 0.0007727
9: -0.0023270, -0.0009135, -0.0024356, -0.0008580, -0.0008960, 0.0009577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008265, upper bound: 0.0006946
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006421, upper bound: 0.0005585
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9938385, 0.9959711, 0.9937639, 0.9961584, -0.0014868, 0.0013567
1: -0.0027992, -0.0022678, -0.0028178, -0.0022212, -0.0003705, 0.0003380
2: 0.0019644, 0.0047804, 0.0017171, 0.0048790, -0.0017915, 0.0019633
3: -0.0034490, -0.0021672, -0.0034938, -0.0020547, -0.0008936, 0.0008154
4: 0.0009081, 0.0014531, 0.0008602, 0.0014722, -0.0003467, 0.0003800
5: 0.0014301, 0.0049720, 0.0011192, 0.0050960, -0.0022532, 0.0024693
6: 0.0002789, 0.0011778, 0.0002474, 0.0012568, -0.0006267, 0.0005719
7: -0.0024161, -0.0000902, -0.0024975, 0.0001140, -0.0016216, 0.0014796
8: -0.0008347, 0.0003884, -0.0008775, 0.0004958, -0.0008528, 0.0007781
9: -0.0023143, -0.0008959, -0.0024388, -0.0008463, -0.0009023, 0.0009888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009038, upper bound: 0.0007976
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008774, upper bound: 0.0007532
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9938560, 0.9961986, 0.9938651, 0.9959879, -0.0012756, 0.0014857
1: -0.0027949, -0.0022112, -0.0027926, -0.0022637, -0.0003178, 0.0003702
2: 0.0016640, 0.0047573, 0.0019422, 0.0047453, -0.0019618, 0.0016844
3: -0.0034385, -0.0020305, -0.0034330, -0.0021571, -0.0007667, 0.0008929
4: 0.0008500, 0.0014487, 0.0009038, 0.0014463, -0.0003797, 0.0003260
5: 0.0010523, 0.0049429, 0.0014022, 0.0049278, -0.0024675, 0.0021185
6: 0.0002863, 0.0012737, 0.0002901, 0.0011849, -0.0005377, 0.0006263
7: -0.0023970, 0.0001579, -0.0023871, -0.0000718, -0.0013912, 0.0016204
8: -0.0008247, 0.0005189, -0.0008195, 0.0003981, -0.0007316, 0.0008521
9: -0.0024656, -0.0009076, -0.0023254, -0.0009136, -0.0009881, 0.0008483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006253, upper bound: 0.0006353
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009832
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009832
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9938185, 0.9961754, 0.9938651, 0.9959879, -0.0013252, 0.0014776
1: -0.0028043, -0.0022169, -0.0027926, -0.0022637, -0.0003302, 0.0003682
2: 0.0016946, 0.0048071, 0.0019422, 0.0047453, -0.0019512, 0.0017499
3: -0.0034611, -0.0020444, -0.0034330, -0.0021571, -0.0007965, 0.0008881
4: 0.0008559, 0.0014583, 0.0009038, 0.0014463, -0.0003777, 0.0003387
5: 0.0010908, 0.0050055, 0.0014022, 0.0049278, -0.0024541, 0.0022009
6: 0.0002704, 0.0012640, 0.0002901, 0.0011849, -0.0005586, 0.0006229
7: -0.0024380, 0.0001327, -0.0023871, -0.0000718, -0.0014453, 0.0016116
8: -0.0008463, 0.0005056, -0.0008195, 0.0003981, -0.0007601, 0.0008475
9: -0.0024501, -0.0008825, -0.0023254, -0.0009136, -0.0009827, 0.0008813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006253, upper bound: 0.0006368
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009831
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008892, upper bound: 0.0009831
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9938353, 0.9961774, 0.9938492, 0.9959928, -0.0013104, 0.0014886
1: -0.0028001, -0.0022165, -0.0027966, -0.0022625, -0.0003265, 0.0003709
2: 0.0016921, 0.0047848, 0.0019359, 0.0047663, -0.0019656, 0.0017303
3: -0.0034510, -0.0020433, -0.0034426, -0.0021542, -0.0007876, 0.0008947
4: 0.0008554, 0.0014540, 0.0009026, 0.0014504, -0.0003804, 0.0003349
5: 0.0010877, 0.0049775, 0.0013942, 0.0049543, -0.0024723, 0.0021763
6: 0.0002775, 0.0012648, 0.0002834, 0.0011870, -0.0005524, 0.0006275
7: -0.0024197, 0.0001347, -0.0024044, -0.0000666, -0.0014291, 0.0016235
8: -0.0008366, 0.0005067, -0.0008286, 0.0004008, -0.0007516, 0.0008538
9: -0.0024514, -0.0008937, -0.0023286, -0.0009030, -0.0009900, 0.0008715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007298, upper bound: 0.0008078
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006236, upper bound: 0.0007293
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9937955, 0.9961548, 0.9938492, 0.9959928, -0.0013628, 0.0014809
1: -0.0028100, -0.0022221, -0.0027966, -0.0022625, -0.0003396, 0.0003690
2: 0.0017219, 0.0048374, 0.0019359, 0.0047663, -0.0019555, 0.0017996
3: -0.0034749, -0.0020569, -0.0034426, -0.0021542, -0.0008191, 0.0008901
4: 0.0008612, 0.0014642, 0.0009026, 0.0014504, -0.0003785, 0.0003483
5: 0.0011251, 0.0050436, 0.0013942, 0.0049543, -0.0024595, 0.0022634
6: 0.0002607, 0.0012553, 0.0002834, 0.0011870, -0.0005745, 0.0006242
7: -0.0024631, 0.0001101, -0.0024044, -0.0000666, -0.0014863, 0.0016151
8: -0.0008595, 0.0004938, -0.0008286, 0.0004008, -0.0007816, 0.0008494
9: -0.0024364, -0.0008673, -0.0023286, -0.0009030, -0.0009849, 0.0009064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007050, upper bound: 0.0008677
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006236, upper bound: 0.0007562
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938200, 0.9961761, 0.9938649, 0.9959903, -0.0013282, 0.0014740
1: -0.0028038, -0.0022168, -0.0027927, -0.0022631, -0.0003309, 0.0003673
2: 0.0016937, 0.0048049, 0.0019391, 0.0047456, -0.0019464, 0.0017538
3: -0.0034601, -0.0020440, -0.0034331, -0.0021557, -0.0007983, 0.0008859
4: 0.0008557, 0.0014579, 0.0009032, 0.0014464, -0.0003767, 0.0003394
5: 0.0010897, 0.0050027, 0.0013984, 0.0049282, -0.0024481, 0.0022058
6: 0.0002711, 0.0012643, 0.0002900, 0.0011859, -0.0005599, 0.0006213
7: -0.0024362, 0.0001334, -0.0023873, -0.0000693, -0.0014485, 0.0016076
8: -0.0008453, 0.0005060, -0.0008196, 0.0003994, -0.0007618, 0.0008454
9: -0.0024506, -0.0008836, -0.0023270, -0.0009135, -0.0009803, 0.0008833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005758, upper bound: 0.0006258
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005869, upper bound: 0.0007486
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008860, upper bound: 0.0009823
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008860, upper bound: 0.0009907
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938053, 0.9961811, 0.9938385, 0.9959711, -0.0013361, 0.0015207
1: -0.0028075, -0.0022155, -0.0027992, -0.0022678, -0.0003329, 0.0003789
2: 0.0016871, 0.0048242, 0.0019644, 0.0047804, -0.0020080, 0.0017644
3: -0.0034689, -0.0020410, -0.0034490, -0.0021672, -0.0008031, 0.0009140
4: 0.0008544, 0.0014616, 0.0009081, 0.0014531, -0.0003887, 0.0003415
5: 0.0010814, 0.0050270, 0.0014301, 0.0049720, -0.0025256, 0.0022191
6: 0.0002649, 0.0012664, 0.0002789, 0.0011778, -0.0005632, 0.0006410
7: -0.0024522, 0.0001389, -0.0024161, -0.0000902, -0.0014573, 0.0016585
8: -0.0008537, 0.0005089, -0.0008347, 0.0003884, -0.0007664, 0.0008722
9: -0.0024539, -0.0008739, -0.0023143, -0.0008959, -0.0010114, 0.0008886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007428, upper bound: 0.0008336
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006407, upper bound: 0.0007539
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9937815, 0.9961535, 0.9938649, 0.9959903, -0.0013473, 0.0014399
1: -0.0028134, -0.0022224, -0.0027927, -0.0022631, -0.0003357, 0.0003588
2: 0.0017235, 0.0048558, 0.0019391, 0.0047456, -0.0019014, 0.0017790
3: -0.0034833, -0.0020576, -0.0034331, -0.0021557, -0.0008097, 0.0008654
4: 0.0008615, 0.0014677, 0.0009032, 0.0014464, -0.0003680, 0.0003443
5: 0.0011272, 0.0050667, 0.0013984, 0.0049282, -0.0023915, 0.0022376
6: 0.0002548, 0.0012547, 0.0002900, 0.0011859, -0.0005679, 0.0006070
7: -0.0024783, 0.0001088, -0.0023873, -0.0000693, -0.0014694, 0.0015705
8: -0.0008674, 0.0004931, -0.0008196, 0.0003994, -0.0007727, 0.0008259
9: -0.0024356, -0.0008580, -0.0023270, -0.0009135, -0.0009577, 0.0008960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006995, upper bound: 0.0008313
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005646, upper bound: 0.0006419
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9937639, 0.9961584, 0.9938385, 0.9959711, -0.0013567, 0.0014868
1: -0.0028178, -0.0022212, -0.0027992, -0.0022678, -0.0003380, 0.0003705
2: 0.0017171, 0.0048790, 0.0019644, 0.0047804, -0.0019633, 0.0017915
3: -0.0034938, -0.0020547, -0.0034490, -0.0021672, -0.0008154, 0.0008936
4: 0.0008602, 0.0014722, 0.0009081, 0.0014531, -0.0003800, 0.0003467
5: 0.0011192, 0.0050960, 0.0014301, 0.0049720, -0.0024693, 0.0022532
6: 0.0002474, 0.0012568, 0.0002789, 0.0011778, -0.0005719, 0.0006267
7: -0.0024975, 0.0001140, -0.0024161, -0.0000902, -0.0014796, 0.0016216
8: -0.0008775, 0.0004958, -0.0008347, 0.0003884, -0.0007781, 0.0008528
9: -0.0024388, -0.0008463, -0.0023143, -0.0008959, -0.0009888, 0.0009023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007974, upper bound: 0.0009033
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007534, upper bound: 0.0008775
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9938560, 0.9961986, 0.9938200, 0.9961761, -0.0012954, 0.0013509
1: -0.0027949, -0.0022112, -0.0028038, -0.0022168, -0.0003228, 0.0003366
2: 0.0016640, 0.0047573, 0.0016937, 0.0048049, -0.0017838, 0.0017106
3: -0.0034385, -0.0020305, -0.0034601, -0.0020440, -0.0007786, 0.0008119
4: 0.0008500, 0.0014487, 0.0008557, 0.0014579, -0.0003453, 0.0003311
5: 0.0010523, 0.0049429, 0.0010897, 0.0050027, -0.0022436, 0.0021515
6: 0.0002863, 0.0012737, 0.0002711, 0.0012643, -0.0005461, 0.0005694
7: -0.0023970, 0.0001579, -0.0024362, 0.0001334, -0.0014128, 0.0014733
8: -0.0008247, 0.0005189, -0.0008453, 0.0005060, -0.0007430, 0.0007748
9: -0.0024656, -0.0009076, -0.0024506, -0.0008836, -0.0008984, 0.0008615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007056, upper bound: 0.0007277
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008978, upper bound: 0.0009839
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008978, upper bound: 0.0009839
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9938560, 0.9961986, 0.9937815, 0.9961535, -0.0012935, 0.0014040
1: -0.0027949, -0.0022112, -0.0028134, -0.0022224, -0.0003223, 0.0003498
2: 0.0016640, 0.0047573, 0.0017235, 0.0048558, -0.0018540, 0.0017080
3: -0.0034385, -0.0020305, -0.0034833, -0.0020576, -0.0007774, 0.0008439
4: 0.0008500, 0.0014487, 0.0008615, 0.0014677, -0.0003588, 0.0003306
5: 0.0010523, 0.0049429, 0.0011272, 0.0050667, -0.0023318, 0.0021482
6: 0.0002863, 0.0012737, 0.0002548, 0.0012547, -0.0005452, 0.0005918
7: -0.0023970, 0.0001579, -0.0024783, 0.0001088, -0.0014107, 0.0015313
8: -0.0008247, 0.0005189, -0.0008674, 0.0004931, -0.0007419, 0.0008053
9: -0.0024656, -0.0009076, -0.0024356, -0.0008580, -0.0009338, 0.0008602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007056, upper bound: 0.0007719
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008978, upper bound: 0.0009840
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008978, upper bound: 0.0009840
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9938353, 0.9961774, 0.9938053, 0.9961811, -0.0013320, 0.0013595
1: -0.0028001, -0.0022165, -0.0028075, -0.0022155, -0.0003319, 0.0003387
2: 0.0016921, 0.0047848, 0.0016871, 0.0048242, -0.0017952, 0.0017589
3: -0.0034510, -0.0020433, -0.0034689, -0.0020410, -0.0008006, 0.0008171
4: 0.0008554, 0.0014540, 0.0008544, 0.0014616, -0.0003475, 0.0003404
5: 0.0010877, 0.0049775, 0.0010814, 0.0050270, -0.0022578, 0.0022123
6: 0.0002775, 0.0012648, 0.0002649, 0.0012664, -0.0005615, 0.0005731
7: -0.0024197, 0.0001347, -0.0024522, 0.0001389, -0.0014528, 0.0014827
8: -0.0008366, 0.0005067, -0.0008537, 0.0005089, -0.0007640, 0.0007797
9: -0.0024514, -0.0008937, -0.0024539, -0.0008739, -0.0009041, 0.0008859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008015, upper bound: 0.0008723
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007614, upper bound: 0.0008220
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9938353, 0.9961774, 0.9937639, 0.9961584, -0.0013279, 0.0014162
1: -0.0028001, -0.0022165, -0.0028178, -0.0022212, -0.0003309, 0.0003529
2: 0.0016921, 0.0047848, 0.0017171, 0.0048790, -0.0018701, 0.0017534
3: -0.0034510, -0.0020433, -0.0034938, -0.0020547, -0.0007981, 0.0008512
4: 0.0008554, 0.0014540, 0.0008602, 0.0014722, -0.0003619, 0.0003394
5: 0.0010877, 0.0049775, 0.0011192, 0.0050960, -0.0023520, 0.0022054
6: 0.0002775, 0.0012648, 0.0002474, 0.0012568, -0.0005597, 0.0005970
7: -0.0024197, 0.0001347, -0.0024975, 0.0001140, -0.0014482, 0.0015445
8: -0.0008366, 0.0005067, -0.0008775, 0.0004958, -0.0007616, 0.0008123
9: -0.0024514, -0.0008937, -0.0024388, -0.0008463, -0.0009419, 0.0008831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008015, upper bound: 0.0008881
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007614, upper bound: 0.0008432
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9937815, 0.9961535, 0.9938560, 0.9961986, -0.0014040, 0.0012935
1: -0.0028134, -0.0022224, -0.0027949, -0.0022112, -0.0003498, 0.0003223
2: 0.0017235, 0.0048558, 0.0016640, 0.0047573, -0.0017080, 0.0018540
3: -0.0034833, -0.0020576, -0.0034385, -0.0020305, -0.0008439, 0.0007774
4: 0.0008615, 0.0014677, 0.0008500, 0.0014487, -0.0003306, 0.0003588
5: 0.0011272, 0.0050667, 0.0010523, 0.0049429, -0.0021482, 0.0023318
6: 0.0002548, 0.0012547, 0.0002863, 0.0012737, -0.0005918, 0.0005452
7: -0.0024783, 0.0001088, -0.0023970, 0.0001579, -0.0015313, 0.0014107
8: -0.0008674, 0.0004931, -0.0008247, 0.0005189, -0.0008053, 0.0007419
9: -0.0024356, -0.0008580, -0.0024656, -0.0009076, -0.0008602, 0.0009338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007182, upper bound: 0.0007994
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008902, upper bound: 0.0009848
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008902, upper bound: 0.0009942
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9937639, 0.9961584, 0.9938353, 0.9961774, -0.0014162, 0.0013279
1: -0.0028178, -0.0022212, -0.0028001, -0.0022165, -0.0003529, 0.0003309
2: 0.0017171, 0.0048790, 0.0016921, 0.0047848, -0.0017534, 0.0018701
3: -0.0034938, -0.0020547, -0.0034510, -0.0020433, -0.0008512, 0.0007981
4: 0.0008602, 0.0014722, 0.0008554, 0.0014540, -0.0003394, 0.0003619
5: 0.0011192, 0.0050960, 0.0010877, 0.0049775, -0.0022054, 0.0023520
6: 0.0002474, 0.0012568, 0.0002775, 0.0012648, -0.0005970, 0.0005597
7: -0.0024975, 0.0001140, -0.0024197, 0.0001347, -0.0015445, 0.0014482
8: -0.0008775, 0.0004958, -0.0008366, 0.0005067, -0.0008123, 0.0007616
9: -0.0024388, -0.0008463, -0.0024514, -0.0008937, -0.0008831, 0.0009419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008186, upper bound: 0.0009013
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007744, upper bound: 0.0008462
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938185, 0.9961754, 0.9937815, 0.9961535, -0.0013119, 0.0013700
1: -0.0028043, -0.0022169, -0.0028134, -0.0022224, -0.0003269, 0.0003414
2: 0.0016946, 0.0048071, 0.0017235, 0.0048558, -0.0018091, 0.0017323
3: -0.0034611, -0.0020444, -0.0034833, -0.0020576, -0.0007885, 0.0008234
4: 0.0008559, 0.0014583, 0.0008615, 0.0014677, -0.0003501, 0.0003353
5: 0.0010908, 0.0050055, 0.0011272, 0.0050667, -0.0022753, 0.0021788
6: 0.0002704, 0.0012640, 0.0002548, 0.0012547, -0.0005530, 0.0005775
7: -0.0024380, 0.0001327, -0.0024783, 0.0001088, -0.0014308, 0.0014942
8: -0.0008463, 0.0005056, -0.0008674, 0.0004931, -0.0007524, 0.0007858
9: -0.0024501, -0.0008825, -0.0024356, -0.0008580, -0.0009111, 0.0008725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0008063
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006966, upper bound: 0.0007188
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9937955, 0.9961548, 0.9937639, 0.9961584, -0.0013520, 0.0013830
1: -0.0028100, -0.0022221, -0.0028178, -0.0022212, -0.0003369, 0.0003446
2: 0.0017219, 0.0048374, 0.0017171, 0.0048790, -0.0018262, 0.0017853
3: -0.0034749, -0.0020569, -0.0034938, -0.0020547, -0.0008126, 0.0008312
4: 0.0008612, 0.0014642, 0.0008602, 0.0014722, -0.0003535, 0.0003455
5: 0.0011251, 0.0050436, 0.0011192, 0.0050960, -0.0022968, 0.0022455
6: 0.0002607, 0.0012553, 0.0002474, 0.0012568, -0.0005699, 0.0005830
7: -0.0024631, 0.0001101, -0.0024975, 0.0001140, -0.0014746, 0.0015083
8: -0.0008595, 0.0004938, -0.0008775, 0.0004958, -0.0007755, 0.0007932
9: -0.0024364, -0.0008673, -0.0024388, -0.0008463, -0.0009198, 0.0008992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008466, upper bound: 0.0009376
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008359, upper bound: 0.0009226
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9938651, 0.9959879, 0.9938645, 0.9962469, -0.0016629, 0.0013799
1: -0.0027926, -0.0022637, -0.0027928, -0.0021991, -0.0004143, 0.0003438
2: 0.0019422, 0.0047453, 0.0016001, 0.0047462, -0.0018221, 0.0021958
3: -0.0034330, -0.0021571, -0.0034334, -0.0020014, -0.0009994, 0.0008293
4: 0.0009038, 0.0014463, 0.0008376, 0.0014465, -0.0003527, 0.0004250
5: 0.0014022, 0.0049278, 0.0009720, 0.0049289, -0.0022917, 0.0027618
6: 0.0002901, 0.0011849, 0.0002898, 0.0012941, -0.0007010, 0.0005817
7: -0.0023871, -0.0000718, -0.0023878, 0.0002107, -0.0018136, 0.0015049
8: -0.0008195, 0.0003981, -0.0008199, 0.0005467, -0.0009538, 0.0007914
9: -0.0023254, -0.0009136, -0.0024977, -0.0009132, -0.0009177, 0.0011059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006189, upper bound: 0.0006320
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009196
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009206
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9938247, 0.9959700, 0.9938645, 0.9962469, -0.0017198, 0.0013808
1: -0.0028027, -0.0022681, -0.0027928, -0.0021991, -0.0004285, 0.0003441
2: 0.0019658, 0.0047987, 0.0016001, 0.0047462, -0.0018233, 0.0022710
3: -0.0034573, -0.0021679, -0.0034334, -0.0020014, -0.0010337, 0.0008299
4: 0.0009084, 0.0014567, 0.0008376, 0.0014465, -0.0003529, 0.0004395
5: 0.0014319, 0.0049950, 0.0009720, 0.0049289, -0.0022933, 0.0028563
6: 0.0002731, 0.0011774, 0.0002898, 0.0012941, -0.0007250, 0.0005821
7: -0.0024312, -0.0000914, -0.0023878, 0.0002107, -0.0018757, 0.0015060
8: -0.0008427, 0.0003878, -0.0008199, 0.0005467, -0.0009864, 0.0007920
9: -0.0023135, -0.0008867, -0.0024977, -0.0009132, -0.0009183, 0.0011438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 78

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006189, upper bound: 0.0006320
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009196
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009738, upper bound: 0.0009206
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9938492, 0.9959928, 0.9938465, 0.9962219, -0.0016589, 0.0014067
1: -0.0027966, -0.0022625, -0.0027972, -0.0022054, -0.0004133, 0.0003505
2: 0.0019359, 0.0047663, 0.0016333, 0.0047699, -0.0018575, 0.0021905
3: -0.0034426, -0.0021542, -0.0034442, -0.0020165, -0.0009970, 0.0008455
4: 0.0009026, 0.0014504, 0.0008440, 0.0014511, -0.0003595, 0.0004240
5: 0.0013942, 0.0049543, 0.0010137, 0.0049587, -0.0023363, 0.0027551
6: 0.0002834, 0.0011870, 0.0002823, 0.0012835, -0.0006993, 0.0005930
7: -0.0024044, -0.0000666, -0.0024073, 0.0001833, -0.0018092, 0.0015342
8: -0.0008286, 0.0004008, -0.0008301, 0.0005322, -0.0009515, 0.0008068
9: -0.0023286, -0.0009030, -0.0024810, -0.0009013, -0.0009356, 0.0011033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.29 + 598.11 = 601.40 seconds
