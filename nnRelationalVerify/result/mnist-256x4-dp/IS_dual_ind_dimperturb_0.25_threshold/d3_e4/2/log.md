## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0024309


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040193, -0.0036703, -0.0040193, -0.0036703, -0.0002498, 0.0002498)
1: (0.0000707, 0.0020029, 0.0000707, 0.0020029, -0.0013830, 0.0013830)
2: (0.0104914, 0.0148083, 0.0104914, 0.0148083, -0.0030897, 0.0030897)
3: (0.0010941, 0.0029132, 0.0010941, 0.0029132, -0.0013020, 0.0013020)
4: (1.0009950, 1.0080526, 1.0009950, 1.0080526, -0.0050513, 0.0050513)
5: (0.0024350, 0.0038079, 0.0024350, 0.0038079, -0.0009827, 0.0009827)
6: (-0.0106984, -0.0089117, -0.0106984, -0.0089117, -0.0012788, 0.0012788)
7: (-0.0101680, -0.0099401, -0.0101680, -0.0099401, -0.0001631, 0.0001631)
8: (-0.0046968, -0.0034623, -0.0046968, -0.0034623, -0.0008836, 0.0008836)
9: (-0.0008376, 0.0053426, -0.0008376, 0.0053426, -0.0044233, 0.0044233)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.01 + 1.62 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0033991, upper bound: 0.0033991

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030569, upper bound: 0.0030808
time: 0.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030808, upper bound: 0.0030808
time: 0.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.48 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 4, lower bound: -0.0030569, upper bound: 0.0030808
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 4, lower bound: -0.0030808, upper bound: 0.0030808

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040193, -0.0036836, -0.0040193, -0.0036703, -0.0002497, 0.0002326
1: 0.0001440, 0.0020027, 0.0000707, 0.0020029, -0.0012881, 0.0013828
2: 0.0104918, 0.0146445, 0.0104914, 0.0148083, -0.0030893, 0.0028779
3: 0.0011631, 0.0029131, 0.0010941, 0.0029132, -0.0012127, 0.0013018
4: 1.0012628, 1.0080520, 1.0009950, 1.0080526, -0.0047049, 0.0050507
5: 0.0024871, 0.0038078, 0.0024350, 0.0038079, -0.0009153, 0.0009826
6: -0.0106983, -0.0089795, -0.0106984, -0.0089117, -0.0012786, 0.0011911
7: -0.0101680, -0.0099488, -0.0101680, -0.0099401, -0.0001631, 0.0001519
8: -0.0046500, -0.0034625, -0.0046968, -0.0034623, -0.0008230, 0.0008834
9: -0.0008370, 0.0051081, -0.0008376, 0.0053426, -0.0044227, 0.0041200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030276, upper bound: 0.0030276
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030276, upper bound: 0.0030808
time: 0.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0040612, -0.0037108, -0.0040193, -0.0036875, -0.0003086, 0.0002335
1: 0.0002948, 0.0022346, 0.0001657, 0.0020027, -0.0012931, 0.0017089
2: 0.0099739, 0.0143076, 0.0104920, 0.0145960, -0.0038178, 0.0028889
3: 0.0013051, 0.0031313, 0.0011836, 0.0029130, -0.0012174, 0.0016088
4: 1.0018135, 1.0088986, 1.0013421, 1.0080516, -0.0047229, 0.0062417
5: 0.0025942, 0.0039725, 0.0025025, 0.0038078, -0.0009188, 0.0012143
6: -0.0109126, -0.0091189, -0.0106982, -0.0089996, -0.0015802, 0.0011957
7: -0.0101954, -0.0099666, -0.0101680, -0.0099513, -0.0002016, 0.0001525
8: -0.0045537, -0.0033144, -0.0046361, -0.0034625, -0.0008261, 0.0010918
9: -0.0015785, 0.0046258, -0.0008368, 0.0050387, -0.0054657, 0.0041358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027493, upper bound: 0.0027013
time: 0.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027466, upper bound: 0.0027466
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 4, lower bound: -0.0030276, upper bound: 0.0030276
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 4, lower bound: -0.0030276, upper bound: 0.0030808
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 4, lower bound: -0.0027493, upper bound: 0.0027013
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 4, lower bound: -0.0027466, upper bound: 0.0027466

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040193, -0.0036836, -0.0040193, -0.0036836, -0.0002326, 0.0002326
1: 0.0001440, 0.0020027, 0.0001440, 0.0020027, -0.0012880, 0.0012880
2: 0.0104918, 0.0146445, 0.0104918, 0.0146445, -0.0028774, 0.0028774
3: 0.0011631, 0.0029131, 0.0011631, 0.0029131, -0.0012126, 0.0012126
4: 1.0012628, 1.0080520, 1.0012628, 1.0080520, -0.0047043, 0.0047043
5: 0.0024871, 0.0038078, 0.0024871, 0.0038078, -0.0009152, 0.0009152
6: -0.0106983, -0.0089795, -0.0106983, -0.0089795, -0.0011910, 0.0011910
7: -0.0101680, -0.0099488, -0.0101680, -0.0099488, -0.0001519, 0.0001519
8: -0.0046500, -0.0034625, -0.0046500, -0.0034625, -0.0008229, 0.0008228
9: -0.0008370, 0.0051081, -0.0008370, 0.0051081, -0.0041194, 0.0041194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0027352
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0027350
time: 0.61 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040193, -0.0036836, -0.0040612, -0.0037108, -0.0002317, 0.0003011
1: 0.0001440, 0.0020027, 0.0002948, 0.0022346, -0.0016673, 0.0012828
2: 0.0104918, 0.0146445, 0.0099739, 0.0143076, -0.0028659, 0.0037250
3: 0.0011631, 0.0029131, 0.0013051, 0.0031313, -0.0015697, 0.0012077
4: 1.0012628, 1.0080520, 1.0018135, 1.0088986, -0.0060899, 0.0046854
5: 0.0024871, 0.0038078, 0.0025942, 0.0039725, -0.0011847, 0.0009115
6: -0.0106983, -0.0089795, -0.0109126, -0.0091189, -0.0011862, 0.0015418
7: -0.0101680, -0.0099488, -0.0101954, -0.0099666, -0.0001513, 0.0001967
8: -0.0046500, -0.0034625, -0.0045537, -0.0033144, -0.0010652, 0.0008195
9: -0.0008370, 0.0051081, -0.0015785, 0.0046258, -0.0041029, 0.0053328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0027494
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0027466
time: 0.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040612, -0.0037108, -0.0040192, -0.0037018, -0.0002935, 0.0002335
1: 0.0002948, 0.0022346, 0.0002448, 0.0020024, -0.0012928, 0.0016248
2: 0.0099739, 0.0143076, 0.0104926, 0.0144192, -0.0036300, 0.0028882
3: 0.0013051, 0.0031313, 0.0012581, 0.0029127, -0.0012171, 0.0015297
4: 1.0018135, 1.0088986, 1.0016310, 1.0080504, -0.0047218, 0.0059347
5: 0.0025942, 0.0039725, 0.0025587, 0.0038076, -0.0009186, 0.0011545
6: -0.0109126, -0.0091189, -0.0106979, -0.0090728, -0.0015025, 0.0011954
7: -0.0101954, -0.0099666, -0.0101680, -0.0099607, -0.0001917, 0.0001525
8: -0.0045537, -0.0033144, -0.0045856, -0.0034627, -0.0008259, 0.0010381
9: -0.0015785, 0.0046258, -0.0008358, 0.0047855, -0.0051969, 0.0041348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026991
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026991
time: 0.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040611, -0.0037214, -0.0040642, -0.0037099, -0.0003012, 0.0002816
1: 0.0003533, 0.0022342, 0.0002896, 0.0022517, -0.0015590, 0.0016679
2: 0.0099747, 0.0141768, 0.0099356, 0.0143191, -0.0037263, 0.0034831
3: 0.0013602, 0.0031310, 0.0013002, 0.0031475, -0.0014678, 0.0015703
4: 1.0020275, 1.0088973, 1.0017947, 1.0089612, -0.0056944, 0.0060920
5: 0.0026358, 0.0039723, 0.0025905, 0.0039847, -0.0011078, 0.0011851
6: -0.0109123, -0.0091731, -0.0109285, -0.0091142, -0.0015423, 0.0014416
7: -0.0101953, -0.0099735, -0.0101974, -0.0099660, -0.0001967, 0.0001839
8: -0.0045162, -0.0033146, -0.0045569, -0.0033034, -0.0009960, 0.0010656
9: -0.0015773, 0.0044384, -0.0016333, 0.0046423, -0.0053347, 0.0049864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027465
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027466
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0027352
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0027350
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0027494
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0027466
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026991
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026991
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027465
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027466

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040192, -0.0036983, -0.0040193, -0.0036836, -0.0002326, 0.0002168
1: 0.0002255, 0.0020025, 0.0001440, 0.0020027, -0.0012003, 0.0012876
2: 0.0104924, 0.0144624, 0.0104918, 0.0146445, -0.0028767, 0.0026815
3: 0.0012399, 0.0029128, 0.0011631, 0.0029131, -0.0011300, 0.0012123
4: 1.0015606, 1.0080508, 1.0012628, 1.0080520, -0.0043840, 0.0047031
5: 0.0025450, 0.0038076, 0.0024871, 0.0038078, -0.0008529, 0.0009149
6: -0.0106980, -0.0090549, -0.0106983, -0.0089795, -0.0011907, 0.0011099
7: -0.0101680, -0.0099584, -0.0101680, -0.0099488, -0.0001519, 0.0001416
8: -0.0045979, -0.0034626, -0.0046500, -0.0034625, -0.0007668, 0.0008227
9: -0.0008361, 0.0048473, -0.0008370, 0.0051081, -0.0041184, 0.0038389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027932, upper bound: 0.0027932
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027932, upper bound: 0.0029217
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0037049, -0.0040192, -0.0036932, -0.0002804, 0.0002247
1: 0.0002622, 0.0022518, 0.0001970, 0.0020024, -0.0012440, 0.0015524
2: 0.0099354, 0.0143804, 0.0104926, 0.0145261, -0.0034682, 0.0027793
3: 0.0012744, 0.0031475, 0.0012130, 0.0029127, -0.0011712, 0.0014615
4: 1.0016944, 1.0089616, 1.0014564, 1.0080506, -0.0045439, 0.0056700
5: 0.0025710, 0.0039848, 0.0025247, 0.0038076, -0.0008840, 0.0011030
6: -0.0109286, -0.0090888, -0.0106979, -0.0090285, -0.0014354, 0.0011503
7: -0.0101974, -0.0099627, -0.0101680, -0.0099550, -0.0001831, 0.0001467
8: -0.0045745, -0.0033034, -0.0046161, -0.0034627, -0.0007948, 0.0009918
9: -0.0016335, 0.0047300, -0.0008359, 0.0049385, -0.0049651, 0.0039789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029217, upper bound: 0.0027932
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029217, upper bound: 0.0029217
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040192, -0.0036983, -0.0040612, -0.0037108, -0.0002316, 0.0002853
1: 0.0002255, 0.0020025, 0.0002948, 0.0022346, -0.0015796, 0.0012825
2: 0.0104924, 0.0144624, 0.0099739, 0.0143076, -0.0028652, 0.0035291
3: 0.0012399, 0.0029128, 0.0013051, 0.0031313, -0.0014872, 0.0012074
4: 1.0015606, 1.0080508, 1.0018135, 1.0088986, -0.0057697, 0.0046843
5: 0.0025450, 0.0038076, 0.0025942, 0.0039725, -0.0011224, 0.0009113
6: -0.0106980, -0.0090549, -0.0109126, -0.0091189, -0.0011859, 0.0014607
7: -0.0101680, -0.0099584, -0.0101954, -0.0099666, -0.0001513, 0.0001863
8: -0.0045979, -0.0034626, -0.0045537, -0.0033144, -0.0010092, 0.0008194
9: -0.0008361, 0.0048473, -0.0015785, 0.0046258, -0.0041019, 0.0050523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0026991
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0027466
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0037049, -0.0040611, -0.0037214, -0.0002768, 0.0002932
1: 0.0002622, 0.0022518, 0.0003533, 0.0022342, -0.0016234, 0.0015329
2: 0.0099354, 0.0143804, 0.0099747, 0.0141768, -0.0034247, 0.0036269
3: 0.0012744, 0.0031475, 0.0013602, 0.0031310, -0.0015284, 0.0014432
4: 1.0016944, 1.0089616, 1.0020275, 1.0088973, -0.0059296, 0.0055989
5: 0.0025710, 0.0039848, 0.0026358, 0.0039723, -0.0011535, 0.0010892
6: -0.0109286, -0.0090888, -0.0109123, -0.0091731, -0.0014174, 0.0015012
7: -0.0101974, -0.0099627, -0.0101953, -0.0099735, -0.0001808, 0.0001915
8: -0.0045745, -0.0033034, -0.0045162, -0.0033146, -0.0010372, 0.0009793
9: -0.0016335, 0.0047300, -0.0015773, 0.0044384, -0.0049028, 0.0051924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0026991
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0027466
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040611, -0.0037249, -0.0040192, -0.0037018, -0.0002934, 0.0002181
1: 0.0003725, 0.0022343, 0.0002448, 0.0020024, -0.0012076, 0.0016245
2: 0.0099746, 0.0141340, 0.0104926, 0.0144192, -0.0036294, 0.0026980
3: 0.0013782, 0.0031310, 0.0012581, 0.0029127, -0.0011369, 0.0015294
4: 1.0020974, 1.0088975, 1.0016310, 1.0080504, -0.0044109, 0.0059336
5: 0.0026494, 0.0039723, 0.0025587, 0.0038076, -0.0008581, 0.0011543
6: -0.0109124, -0.0091908, -0.0106979, -0.0090728, -0.0015022, 0.0011167
7: -0.0101953, -0.0099757, -0.0101680, -0.0099607, -0.0001916, 0.0001424
8: -0.0045040, -0.0033145, -0.0045856, -0.0034627, -0.0007715, 0.0010379
9: -0.0015775, 0.0043773, -0.0008358, 0.0047855, -0.0051959, 0.0038625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041043, -0.0037352, -0.0040192, -0.0037018, -0.0003392, 0.0002198
1: 0.0004299, 0.0024736, 0.0002448, 0.0020024, -0.0012171, 0.0018782
2: 0.0094399, 0.0140057, 0.0104926, 0.0144192, -0.0041961, 0.0027192
3: 0.0014323, 0.0033564, 0.0012581, 0.0029127, -0.0011459, 0.0017682
4: 1.0023072, 1.0097717, 1.0016310, 1.0080504, -0.0044456, 0.0068601
5: 0.0026902, 0.0041424, 0.0025587, 0.0038076, -0.0008648, 0.0013346
6: -0.0111337, -0.0092439, -0.0106979, -0.0090728, -0.0017367, 0.0011255
7: -0.0102236, -0.0099825, -0.0101680, -0.0099607, -0.0002215, 0.0001436
8: -0.0044673, -0.0031616, -0.0045856, -0.0034627, -0.0007776, 0.0011999
9: -0.0023430, 0.0041935, -0.0008358, 0.0047855, -0.0060072, 0.0038929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040611, -0.0037249, -0.0040642, -0.0037099, -0.0002920, 0.0002734
1: 0.0003725, 0.0022343, 0.0002896, 0.0022517, -0.0015136, 0.0016165
2: 0.0099746, 0.0141340, 0.0099356, 0.0143191, -0.0036115, 0.0033816
3: 0.0013782, 0.0031310, 0.0013002, 0.0031475, -0.0014250, 0.0015219
4: 1.0020974, 1.0088975, 1.0017947, 1.0089612, -0.0055286, 0.0059044
5: 0.0026494, 0.0039723, 0.0025905, 0.0039847, -0.0010755, 0.0011486
6: -0.0109124, -0.0091908, -0.0109285, -0.0091142, -0.0014948, 0.0013996
7: -0.0101953, -0.0099757, -0.0101974, -0.0099660, -0.0001907, 0.0001785
8: -0.0045040, -0.0033145, -0.0045569, -0.0033034, -0.0009670, 0.0010328
9: -0.0015775, 0.0043773, -0.0016333, 0.0046423, -0.0051703, 0.0048412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027349
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027349
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041043, -0.0037352, -0.0040642, -0.0037099, -0.0003037, 0.0002271
1: 0.0004299, 0.0024736, 0.0002896, 0.0022517, -0.0012574, 0.0016813
2: 0.0094399, 0.0140057, 0.0099356, 0.0143191, -0.0037563, 0.0028092
3: 0.0014323, 0.0033564, 0.0013002, 0.0031475, -0.0011838, 0.0015829
4: 1.0023072, 1.0097717, 1.0017947, 1.0089612, -0.0045928, 0.0061410
5: 0.0026902, 0.0041424, 0.0025905, 0.0039847, -0.0008935, 0.0011947
6: -0.0111337, -0.0092439, -0.0109285, -0.0091142, -0.0015547, 0.0011627
7: -0.0102236, -0.0099825, -0.0101974, -0.0099660, -0.0001983, 0.0001483
8: -0.0044673, -0.0031616, -0.0045569, -0.0033034, -0.0008033, 0.0010742
9: -0.0023430, 0.0041935, -0.0016333, 0.0046423, -0.0053775, 0.0040218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026627
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026627
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.49 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027932, upper bound: 0.0027932
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027932, upper bound: 0.0029217
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0029217, upper bound: 0.0027932
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0029217, upper bound: 0.0029217
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0026991
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0026916, upper bound: 0.0027466
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0026991
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027993, upper bound: 0.0027466
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0027245, upper bound: 0.0026630
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027349
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0027349
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026627
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 4, lower bound: -0.0026991, upper bound: 0.0026627

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040192, -0.0036983, -0.0040192, -0.0036983, -0.0002167, 0.0002167
1: 0.0002255, 0.0020025, 0.0002255, 0.0020025, -0.0012000, 0.0012000
2: 0.0104924, 0.0144624, 0.0104924, 0.0144624, -0.0026808, 0.0026808
3: 0.0012399, 0.0029128, 0.0012399, 0.0029128, -0.0011297, 0.0011297
4: 1.0015606, 1.0080508, 1.0015606, 1.0080508, -0.0043828, 0.0043828
5: 0.0025450, 0.0038076, 0.0025450, 0.0038076, -0.0008526, 0.0008526
6: -0.0106980, -0.0090549, -0.0106980, -0.0090549, -0.0011096, 0.0011096
7: -0.0101680, -0.0099584, -0.0101680, -0.0099584, -0.0001415, 0.0001415
8: -0.0045979, -0.0034626, -0.0045979, -0.0034626, -0.0007666, 0.0007666
9: -0.0008361, 0.0048473, -0.0008361, 0.0048473, -0.0038380, 0.0038380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025491, upper bound: 0.0025160
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025035, upper bound: 0.0025523
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040192, -0.0036983, -0.0040643, -0.0037049, -0.0002186, 0.0002720
1: 0.0002255, 0.0020025, 0.0002622, 0.0022518, -0.0015060, 0.0012106
2: 0.0104924, 0.0144624, 0.0099354, 0.0143804, -0.0027046, 0.0033645
3: 0.0012399, 0.0029128, 0.0012744, 0.0031475, -0.0014178, 0.0011397
4: 1.0015606, 1.0080508, 1.0016944, 1.0089616, -0.0055005, 0.0044217
5: 0.0025450, 0.0038076, 0.0025710, 0.0039848, -0.0010701, 0.0008602
6: -0.0106980, -0.0090549, -0.0109286, -0.0090888, -0.0011194, 0.0013925
7: -0.0101680, -0.0099584, -0.0101974, -0.0099627, -0.0001428, 0.0001776
8: -0.0045979, -0.0034626, -0.0045745, -0.0033034, -0.0009621, 0.0007734
9: -0.0008361, 0.0048473, -0.0016335, 0.0047300, -0.0038719, 0.0048166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025491, upper bound: 0.0025443
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025035, upper bound: 0.0025649
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0037049, -0.0040192, -0.0036983, -0.0002720, 0.0002186
1: 0.0002622, 0.0022518, 0.0002255, 0.0020025, -0.0012106, 0.0015060
2: 0.0099354, 0.0143804, 0.0104924, 0.0144624, -0.0033645, 0.0027046
3: 0.0012744, 0.0031475, 0.0012399, 0.0029128, -0.0011397, 0.0014178
4: 1.0016944, 1.0089616, 1.0015606, 1.0080508, -0.0044217, 0.0055005
5: 0.0025710, 0.0039848, 0.0025450, 0.0038076, -0.0008602, 0.0010701
6: -0.0109286, -0.0090888, -0.0106980, -0.0090549, -0.0013925, 0.0011194
7: -0.0101974, -0.0099627, -0.0101680, -0.0099584, -0.0001776, 0.0001428
8: -0.0045745, -0.0033034, -0.0045979, -0.0034626, -0.0007734, 0.0009621
9: -0.0016335, 0.0047300, -0.0008361, 0.0048473, -0.0048166, 0.0038719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026386, upper bound: 0.0024777
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025529, upper bound: 0.0025017
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0037049, -0.0040643, -0.0037049, -0.0002273, 0.0002273
1: 0.0002622, 0.0022518, 0.0002622, 0.0022518, -0.0012584, 0.0012584
2: 0.0099354, 0.0143804, 0.0099354, 0.0143804, -0.0028114, 0.0028114
3: 0.0012744, 0.0031475, 0.0012744, 0.0031475, -0.0011847, 0.0011847
4: 1.0016944, 1.0089616, 1.0016944, 1.0089616, -0.0045963, 0.0045963
5: 0.0025710, 0.0039848, 0.0025710, 0.0039848, -0.0008942, 0.0008942
6: -0.0109286, -0.0090888, -0.0109286, -0.0090888, -0.0011636, 0.0011636
7: -0.0101974, -0.0099627, -0.0101974, -0.0099627, -0.0001484, 0.0001484
8: -0.0045745, -0.0033034, -0.0045745, -0.0033034, -0.0008040, 0.0008040
9: -0.0016335, 0.0047300, -0.0016335, 0.0047300, -0.0040249, 0.0040249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026386, upper bound: 0.0024777
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025529, upper bound: 0.0025017
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040192, -0.0036983, -0.0040611, -0.0037249, -0.0002172, 0.0002852
1: 0.0002255, 0.0020025, 0.0003725, 0.0022343, -0.0015793, 0.0012026
2: 0.0104924, 0.0144624, 0.0099746, 0.0141340, -0.0026868, 0.0035284
3: 0.0012399, 0.0029128, 0.0013782, 0.0031310, -0.0014869, 0.0011322
4: 1.0015606, 1.0080508, 1.0020974, 1.0088975, -0.0057685, 0.0043927
5: 0.0025450, 0.0038076, 0.0026494, 0.0039723, -0.0011222, 0.0008545
6: -0.0106980, -0.0090549, -0.0109124, -0.0091908, -0.0011121, 0.0014604
7: -0.0101680, -0.0099584, -0.0101953, -0.0099757, -0.0001419, 0.0001863
8: -0.0045979, -0.0034626, -0.0045040, -0.0033145, -0.0010090, 0.0007683
9: -0.0008361, 0.0048473, -0.0015775, 0.0043773, -0.0038466, 0.0050514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024163, upper bound: 0.0023294
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023353, upper bound: 0.0023278
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040192, -0.0036983, -0.0041043, -0.0037352, -0.0002102, 0.0003310
1: 0.0002255, 0.0020025, 0.0004299, 0.0024736, -0.0018330, 0.0011640
2: 0.0104924, 0.0144624, 0.0094399, 0.0140057, -0.0026005, 0.0040951
3: 0.0012399, 0.0029128, 0.0014323, 0.0033564, -0.0017257, 0.0010959
4: 1.0015606, 1.0080508, 1.0023072, 1.0097717, -0.0066951, 0.0042516
5: 0.0025450, 0.0038076, 0.0026902, 0.0041424, -0.0013025, 0.0008271
6: -0.0106980, -0.0090549, -0.0111337, -0.0092439, -0.0010763, 0.0016950
7: -0.0101680, -0.0099584, -0.0102236, -0.0099825, -0.0001373, 0.0002162
8: -0.0045979, -0.0034626, -0.0044673, -0.0031616, -0.0011711, 0.0007437
9: -0.0008361, 0.0048473, -0.0023430, 0.0041935, -0.0037230, 0.0058627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024163, upper bound: 0.0023294
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023353, upper bound: 0.0023278
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0037049, -0.0040611, -0.0037249, -0.0002725, 0.0002872
1: 0.0002622, 0.0022518, 0.0003725, 0.0022343, -0.0015900, 0.0015086
2: 0.0099354, 0.0143804, 0.0099746, 0.0141340, -0.0033705, 0.0035521
3: 0.0012744, 0.0031475, 0.0013782, 0.0031310, -0.0014969, 0.0014203
4: 1.0016944, 1.0089616, 1.0020974, 1.0088975, -0.0058073, 0.0055103
5: 0.0025710, 0.0039848, 0.0026494, 0.0039723, -0.0011298, 0.0010720
6: -0.0109286, -0.0090888, -0.0109124, -0.0091908, -0.0013950, 0.0014702
7: -0.0101974, -0.0099627, -0.0101953, -0.0099757, -0.0001779, 0.0001875
8: -0.0045745, -0.0033034, -0.0045040, -0.0033145, -0.0010158, 0.0009638
9: -0.0016335, 0.0047300, -0.0015775, 0.0043773, -0.0048252, 0.0050853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024718, upper bound: 0.0022846
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023594, upper bound: 0.0022665
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0037049, -0.0041043, -0.0037352, -0.0002272, 0.0002956
1: 0.0002622, 0.0022518, 0.0004299, 0.0024736, -0.0016368, 0.0012581
2: 0.0099354, 0.0143804, 0.0094399, 0.0140057, -0.0028107, 0.0036569
3: 0.0012744, 0.0031475, 0.0014323, 0.0033564, -0.0015410, 0.0011844
4: 1.0016944, 1.0089616, 1.0023072, 1.0097717, -0.0059786, 0.0045951
5: 0.0025710, 0.0039848, 0.0026902, 0.0041424, -0.0011631, 0.0008939
6: -0.0109286, -0.0090888, -0.0111337, -0.0092439, -0.0011633, 0.0015136
7: -0.0101974, -0.0099627, -0.0102236, -0.0099825, -0.0001484, 0.0001931
8: -0.0045745, -0.0033034, -0.0044673, -0.0031616, -0.0010458, 0.0008038
9: -0.0016335, 0.0047300, -0.0023430, 0.0041935, -0.0040238, 0.0052353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024718, upper bound: 0.0022846
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023594, upper bound: 0.0022665
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040611, -0.0037249, -0.0040192, -0.0036983, -0.0002852, 0.0002172
1: 0.0003725, 0.0022343, 0.0002255, 0.0020025, -0.0012026, 0.0015793
2: 0.0099746, 0.0141340, 0.0104924, 0.0144624, -0.0035284, 0.0026868
3: 0.0013782, 0.0031310, 0.0012399, 0.0029128, -0.0011322, 0.0014869
4: 1.0020974, 1.0088975, 1.0015606, 1.0080508, -0.0043927, 0.0057685
5: 0.0026494, 0.0039723, 0.0025450, 0.0038076, -0.0008545, 0.0011222
6: -0.0109124, -0.0091908, -0.0106980, -0.0090549, -0.0014604, 0.0011121
7: -0.0101953, -0.0099757, -0.0101680, -0.0099584, -0.0001863, 0.0001419
8: -0.0045040, -0.0033145, -0.0045979, -0.0034626, -0.0007683, 0.0010090
9: -0.0015775, 0.0043773, -0.0008361, 0.0048473, -0.0050514, 0.0038466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025758, upper bound: 0.0024862
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025619, upper bound: 0.0025395
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040611, -0.0037249, -0.0040611, -0.0037249, -0.0002221, 0.0002221
1: 0.0003725, 0.0022343, 0.0003725, 0.0022343, -0.0012297, 0.0012297
2: 0.0099746, 0.0141340, 0.0099746, 0.0141340, -0.0027473, 0.0027473
3: 0.0013782, 0.0031310, 0.0013782, 0.0031310, -0.0011577, 0.0011577
4: 1.0020974, 1.0088975, 1.0020974, 1.0088975, -0.0044915, 0.0044915
5: 0.0026494, 0.0039723, 0.0026494, 0.0039723, -0.0008738, 0.0008738
6: -0.0109124, -0.0091908, -0.0109124, -0.0091908, -0.0011371, 0.0011371
7: -0.0101953, -0.0099757, -0.0101953, -0.0099757, -0.0001450, 0.0001450
8: -0.0045040, -0.0033145, -0.0045040, -0.0033145, -0.0007856, 0.0007856
9: -0.0015775, 0.0043773, -0.0015775, 0.0043773, -0.0039331, 0.0039331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025758, upper bound: 0.0024862
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025619, upper bound: 0.0025395
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041043, -0.0037352, -0.0040192, -0.0036983, -0.0003310, 0.0002102
1: 0.0004299, 0.0024736, 0.0002255, 0.0020025, -0.0011640, 0.0018330
2: 0.0094399, 0.0140057, 0.0104924, 0.0144624, -0.0040951, 0.0026005
3: 0.0014323, 0.0033564, 0.0012399, 0.0029128, -0.0010959, 0.0017257
4: 1.0023072, 1.0097717, 1.0015606, 1.0080508, -0.0042516, 0.0066951
5: 0.0026902, 0.0041424, 0.0025450, 0.0038076, -0.0008271, 0.0013025
6: -0.0111337, -0.0092439, -0.0106980, -0.0090549, -0.0016950, 0.0010763
7: -0.0102236, -0.0099825, -0.0101680, -0.0099584, -0.0002162, 0.0001373
8: -0.0044673, -0.0031616, -0.0045979, -0.0034626, -0.0007437, 0.0011711
9: -0.0023430, 0.0041935, -0.0008361, 0.0048473, -0.0058627, 0.0037230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024201, upper bound: 0.0023032
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023263, upper bound: 0.0022801
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041043, -0.0037352, -0.0040611, -0.0037249, -0.0002776, 0.0002238
1: 0.0004299, 0.0024736, 0.0003725, 0.0022343, -0.0012392, 0.0015372
2: 0.0094399, 0.0140057, 0.0099746, 0.0141340, -0.0034343, 0.0027685
3: 0.0014323, 0.0033564, 0.0013782, 0.0031310, -0.0011666, 0.0014472
4: 1.0023072, 1.0097717, 1.0020974, 1.0088975, -0.0045262, 0.0056147
5: 0.0026902, 0.0041424, 0.0026494, 0.0039723, -0.0008805, 0.0010923
6: -0.0111337, -0.0092439, -0.0109124, -0.0091908, -0.0014214, 0.0011459
7: -0.0102236, -0.0099825, -0.0101953, -0.0099757, -0.0001813, 0.0001462
8: -0.0044673, -0.0031616, -0.0045040, -0.0033145, -0.0007917, 0.0009821
9: -0.0023430, 0.0041935, -0.0015775, 0.0043773, -0.0049167, 0.0039634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024201, upper bound: 0.0023032
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023263, upper bound: 0.0022801
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040611, -0.0037249, -0.0040643, -0.0037049, -0.0002872, 0.0002725
1: 0.0003725, 0.0022343, 0.0002622, 0.0022518, -0.0015086, 0.0015900
2: 0.0099746, 0.0141340, 0.0099354, 0.0143804, -0.0035522, 0.0033705
3: 0.0013782, 0.0031310, 0.0012744, 0.0031475, -0.0014203, 0.0014969
4: 1.0020974, 1.0088975, 1.0016944, 1.0089616, -0.0055103, 0.0058073
5: 0.0026494, 0.0039723, 0.0025710, 0.0039848, -0.0010720, 0.0011298
6: -0.0109124, -0.0091908, -0.0109286, -0.0090888, -0.0014702, 0.0013950
7: -0.0101953, -0.0099757, -0.0101974, -0.0099627, -0.0001875, 0.0001779
8: -0.0045040, -0.0033145, -0.0045745, -0.0033034, -0.0009638, 0.0010158
9: -0.0015775, 0.0043773, -0.0016335, 0.0047300, -0.0050853, 0.0048252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023620, upper bound: 0.0023249
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022833, upper bound: 0.0023245
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040611, -0.0037249, -0.0041043, -0.0037352, -0.0002238, 0.0002776
1: 0.0003725, 0.0022343, 0.0004299, 0.0024736, -0.0015372, 0.0012392
2: 0.0099746, 0.0141340, 0.0094399, 0.0140057, -0.0027685, 0.0034343
3: 0.0013782, 0.0031310, 0.0014323, 0.0033564, -0.0014472, 0.0011666
4: 1.0020974, 1.0088975, 1.0023072, 1.0097717, -0.0056147, 0.0045262
5: 0.0026494, 0.0039723, 0.0026902, 0.0041424, -0.0010923, 0.0008805
6: -0.0109124, -0.0091908, -0.0111337, -0.0092439, -0.0011459, 0.0014214
7: -0.0101953, -0.0099757, -0.0102236, -0.0099825, -0.0001462, 0.0001813
8: -0.0045040, -0.0033145, -0.0044673, -0.0031616, -0.0009821, 0.0007917
9: -0.0015775, 0.0043773, -0.0023430, 0.0041935, -0.0039634, 0.0049167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023620, upper bound: 0.0023249
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022833, upper bound: 0.0023245
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041043, -0.0037352, -0.0040643, -0.0037049, -0.0002956, 0.0002272
1: 0.0004299, 0.0024736, 0.0002622, 0.0022518, -0.0012581, 0.0016368
2: 0.0094399, 0.0140057, 0.0099354, 0.0143804, -0.0036569, 0.0028107
3: 0.0014323, 0.0033564, 0.0012744, 0.0031475, -0.0011844, 0.0015410
4: 1.0023072, 1.0097717, 1.0016944, 1.0089616, -0.0045951, 0.0059786
5: 0.0026902, 0.0041424, 0.0025710, 0.0039848, -0.0008939, 0.0011631
6: -0.0111337, -0.0092439, -0.0109286, -0.0090888, -0.0015136, 0.0011633
7: -0.0102236, -0.0099825, -0.0101974, -0.0099627, -0.0001931, 0.0001484
8: -0.0044673, -0.0031616, -0.0045745, -0.0033034, -0.0008038, 0.0010458
9: -0.0023430, 0.0041935, -0.0016335, 0.0047300, -0.0052353, 0.0040238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023783, upper bound: 0.0022800
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022588, upper bound: 0.0022556
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041043, -0.0037352, -0.0041043, -0.0037352, -0.0002312, 0.0002312
1: 0.0004299, 0.0024736, 0.0004299, 0.0024736, -0.0012800, 0.0012800
2: 0.0094399, 0.0140057, 0.0094399, 0.0140057, -0.0028596, 0.0028596
3: 0.0014323, 0.0033564, 0.0014323, 0.0033564, -0.0012050, 0.0012050
4: 1.0023072, 1.0097717, 1.0023072, 1.0097717, -0.0046751, 0.0046751
5: 0.0026902, 0.0041424, 0.0026902, 0.0041424, -0.0009095, 0.0009095
6: -0.0111337, -0.0092439, -0.0111337, -0.0092439, -0.0011836, 0.0011836
7: -0.0102236, -0.0099825, -0.0102236, -0.0099825, -0.0001510, 0.0001510
8: -0.0044673, -0.0031616, -0.0044673, -0.0031616, -0.0008177, 0.0008177
9: -0.0023430, 0.0041935, -0.0023430, 0.0041935, -0.0040938, 0.0040938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023783, upper bound: 0.0022800
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022588, upper bound: 0.0022556
time: 0.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025491, upper bound: 0.0025160
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025035, upper bound: 0.0025523
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025491, upper bound: 0.0025443
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025035, upper bound: 0.0025649
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0026386, upper bound: 0.0024777
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025529, upper bound: 0.0025017
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0026386, upper bound: 0.0024777
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025529, upper bound: 0.0025017
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0024163, upper bound: 0.0023294
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023353, upper bound: 0.0023278
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0024163, upper bound: 0.0023294
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023353, upper bound: 0.0023278
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0024718, upper bound: 0.0022846
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023594, upper bound: 0.0022665
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0024718, upper bound: 0.0022846
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023594, upper bound: 0.0022665
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025758, upper bound: 0.0024862
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025619, upper bound: 0.0025395
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025758, upper bound: 0.0024862
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0025619, upper bound: 0.0025395
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0024201, upper bound: 0.0023032
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023263, upper bound: 0.0022801
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0024201, upper bound: 0.0023032
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023263, upper bound: 0.0022801
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023620, upper bound: 0.0023249
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0022833, upper bound: 0.0023245
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023620, upper bound: 0.0023249
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0022833, upper bound: 0.0023245
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023783, upper bound: 0.0022800
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0022588, upper bound: 0.0022556
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0023783, upper bound: 0.0022800
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -0.0022588, upper bound: 0.0022556

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040125, -0.0036996, -0.0040192, -0.0036983, -0.0002107, 0.0002157
1: 0.0002329, 0.0019650, 0.0002255, 0.0020025, -0.0011944, 0.0011667
2: 0.0105762, 0.0144458, 0.0104924, 0.0144624, -0.0026065, 0.0026684
3: 0.0012468, 0.0028775, 0.0012399, 0.0029128, -0.0011245, 0.0010984
4: 1.0015875, 1.0079141, 1.0015606, 1.0080508, -0.0043626, 0.0042614
5: 0.0025502, 0.0037810, 0.0025450, 0.0038076, -0.0008487, 0.0008290
6: -0.0106634, -0.0090617, -0.0106980, -0.0090549, -0.0010788, 0.0011045
7: -0.0101636, -0.0099593, -0.0101680, -0.0099584, -0.0001376, 0.0001409
8: -0.0045932, -0.0034866, -0.0045979, -0.0034626, -0.0007631, 0.0007454
9: -0.0007162, 0.0048237, -0.0008361, 0.0048473, -0.0037316, 0.0038202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025933, upper bound: 0.0025933
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025933, upper bound: 0.0025932
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036651, -0.0040146, -0.0036999, -0.0002088, 0.0002515
1: 0.0000417, 0.0019349, 0.0002346, 0.0019766, -0.0013925, 0.0011562
2: 0.0106433, 0.0148730, 0.0105503, 0.0144421, -0.0025831, 0.0031110
3: 0.0010669, 0.0028492, 0.0012484, 0.0028884, -0.0013110, 0.0010885
4: 1.0008893, 1.0078043, 1.0015936, 1.0079563, -0.0050861, 0.0042231
5: 0.0024144, 0.0037596, 0.0025514, 0.0037892, -0.0009895, 0.0008216
6: -0.0106356, -0.0088849, -0.0106741, -0.0090633, -0.0010691, 0.0012876
7: -0.0101600, -0.0099367, -0.0101649, -0.0099595, -0.0001364, 0.0001642
8: -0.0047153, -0.0035058, -0.0045921, -0.0034792, -0.0008896, 0.0007387
9: -0.0006201, 0.0054351, -0.0007533, 0.0048183, -0.0036980, 0.0044538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025145, upper bound: 0.0025159
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025612, upper bound: 0.0025612
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040125, -0.0036996, -0.0040643, -0.0037049, -0.0002126, 0.0002710
1: 0.0002329, 0.0019650, 0.0002622, 0.0022518, -0.0015004, 0.0011773
2: 0.0105762, 0.0144458, 0.0099354, 0.0143804, -0.0026303, 0.0033521
3: 0.0012468, 0.0028775, 0.0012744, 0.0031475, -0.0014126, 0.0011084
4: 1.0015875, 1.0079141, 1.0016944, 1.0089616, -0.0054802, 0.0043002
5: 0.0025502, 0.0037810, 0.0025710, 0.0039848, -0.0010661, 0.0008366
6: -0.0106634, -0.0090617, -0.0109286, -0.0090888, -0.0010887, 0.0013874
7: -0.0101636, -0.0099593, -0.0101974, -0.0099627, -0.0001389, 0.0001770
8: -0.0045932, -0.0034866, -0.0045745, -0.0033034, -0.0009586, 0.0007522
9: -0.0007162, 0.0048237, -0.0016335, 0.0047300, -0.0037656, 0.0047989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024806, upper bound: 0.0025433
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024806, upper bound: 0.0025433
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036651, -0.0040593, -0.0037066, -0.0002106, 0.0003063
1: 0.0000417, 0.0019349, 0.0002713, 0.0022243, -0.0016960, 0.0011662
2: 0.0106433, 0.0148730, 0.0099969, 0.0143600, -0.0026055, 0.0037891
3: 0.0010669, 0.0028492, 0.0012830, 0.0031216, -0.0015968, 0.0010979
4: 1.0008893, 1.0078043, 1.0017278, 1.0088609, -0.0061948, 0.0042596
5: 0.0024144, 0.0037596, 0.0025775, 0.0039652, -0.0012051, 0.0008287
6: -0.0106356, -0.0088849, -0.0109031, -0.0090972, -0.0010784, 0.0015683
7: -0.0101600, -0.0099367, -0.0101942, -0.0099638, -0.0001376, 0.0002001
8: -0.0047153, -0.0035058, -0.0045686, -0.0033209, -0.0010836, 0.0007451
9: -0.0006201, 0.0054351, -0.0015455, 0.0047008, -0.0037300, 0.0054246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022964, upper bound: 0.0023486
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0023971, upper bound: 0.0024582
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0040192, -0.0036983, -0.0002658, 0.0002176
1: 0.0002694, 0.0022149, 0.0002255, 0.0020025, -0.0012049, 0.0014715
2: 0.0100178, 0.0143643, 0.0104924, 0.0144624, -0.0032876, 0.0026918
3: 0.0012812, 0.0031128, 0.0012399, 0.0029128, -0.0011343, 0.0013854
4: 1.0017209, 1.0088270, 1.0015606, 1.0080508, -0.0044008, 0.0053748
5: 0.0025762, 0.0039586, 0.0025450, 0.0038076, -0.0008561, 0.0010456
6: -0.0108945, -0.0090955, -0.0106980, -0.0090549, -0.0013607, 0.0011141
7: -0.0101931, -0.0099636, -0.0101680, -0.0099584, -0.0001736, 0.0001421
8: -0.0045699, -0.0033269, -0.0045979, -0.0034626, -0.0007698, 0.0009401
9: -0.0015156, 0.0047070, -0.0008361, 0.0048473, -0.0047066, 0.0038537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025433, upper bound: 0.0024806
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025433, upper bound: 0.0024806
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040507, -0.0036735, -0.0040146, -0.0036999, -0.0002640, 0.0002523
1: 0.0000881, 0.0021765, 0.0002346, 0.0019766, -0.0013972, 0.0014617
2: 0.0101037, 0.0147694, 0.0105503, 0.0144421, -0.0032657, 0.0031214
3: 0.0011105, 0.0030766, 0.0012484, 0.0028884, -0.0013154, 0.0013762
4: 1.0010585, 1.0086864, 1.0015936, 1.0079563, -0.0051031, 0.0053390
5: 0.0024473, 0.0039313, 0.0025514, 0.0037892, -0.0009928, 0.0010387
6: -0.0108589, -0.0089278, -0.0106741, -0.0090633, -0.0013517, 0.0012919
7: -0.0101885, -0.0099422, -0.0101649, -0.0099595, -0.0001724, 0.0001648
8: -0.0046857, -0.0033515, -0.0045921, -0.0034792, -0.0008926, 0.0009339
9: -0.0013927, 0.0052869, -0.0007533, 0.0048183, -0.0046753, 0.0044687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023406, upper bound: 0.0023064
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024582, upper bound: 0.0023970
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0040643, -0.0037049, -0.0002212, 0.0002263
1: 0.0002694, 0.0022149, 0.0002622, 0.0022518, -0.0012529, 0.0012249
2: 0.0100178, 0.0143643, 0.0099354, 0.0143804, -0.0027365, 0.0027990
3: 0.0012812, 0.0031128, 0.0012744, 0.0031475, -0.0011795, 0.0011532
4: 1.0017209, 1.0088270, 1.0016944, 1.0089616, -0.0045761, 0.0044738
5: 0.0025762, 0.0039586, 0.0025710, 0.0039848, -0.0008902, 0.0008703
6: -0.0108945, -0.0090955, -0.0109286, -0.0090888, -0.0011326, 0.0011585
7: -0.0101931, -0.0099636, -0.0101974, -0.0099627, -0.0001445, 0.0001478
8: -0.0045699, -0.0033269, -0.0045745, -0.0033034, -0.0008004, 0.0007825
9: -0.0015156, 0.0047070, -0.0016335, 0.0047300, -0.0039176, 0.0040072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025403, upper bound: 0.0024775
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025403, upper bound: 0.0024775
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040507, -0.0036735, -0.0040593, -0.0037066, -0.0002195, 0.0002623
1: 0.0000881, 0.0021765, 0.0002713, 0.0022243, -0.0014526, 0.0012152
2: 0.0101037, 0.0147694, 0.0099969, 0.0143600, -0.0027149, 0.0032452
3: 0.0011105, 0.0030766, 0.0012830, 0.0031216, -0.0013675, 0.0011441
4: 1.0010585, 1.0086864, 1.0017278, 1.0088609, -0.0053055, 0.0044385
5: 0.0024473, 0.0039313, 0.0025775, 0.0039652, -0.0010321, 0.0008635
6: -0.0108589, -0.0089278, -0.0109031, -0.0090972, -0.0011237, 0.0013432
7: -0.0101885, -0.0099422, -0.0101942, -0.0099638, -0.0001433, 0.0001713
8: -0.0046857, -0.0033515, -0.0045686, -0.0033209, -0.0009280, 0.0007764
9: -0.0013927, 0.0052869, -0.0015455, 0.0047008, -0.0038867, 0.0046459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023176, upper bound: 0.0023020
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024461, upper bound: 0.0023934
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0040611, -0.0037249, -0.0002663, 0.0002861
1: 0.0002694, 0.0022149, 0.0003725, 0.0022343, -0.0015842, 0.0014742
2: 0.0100178, 0.0143643, 0.0099746, 0.0141340, -0.0032936, 0.0035394
3: 0.0012812, 0.0031128, 0.0013782, 0.0031310, -0.0014915, 0.0013879
4: 1.0017209, 1.0088270, 1.0020974, 1.0088975, -0.0057865, 0.0053846
5: 0.0025762, 0.0039586, 0.0026494, 0.0039723, -0.0011257, 0.0010475
6: -0.0108945, -0.0090955, -0.0109124, -0.0091908, -0.0013632, 0.0014649
7: -0.0101931, -0.0099636, -0.0101953, -0.0099757, -0.0001739, 0.0001869
8: -0.0045699, -0.0033269, -0.0045040, -0.0033145, -0.0010121, 0.0009419
9: -0.0015156, 0.0047070, -0.0015775, 0.0043773, -0.0047152, 0.0050671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023869, upper bound: 0.0022890
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023869, upper bound: 0.0022890
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0041043, -0.0037352, -0.0002212, 0.0002946
1: 0.0002694, 0.0022149, 0.0004299, 0.0024736, -0.0016313, 0.0012245
2: 0.0100178, 0.0143643, 0.0094399, 0.0140057, -0.0027357, 0.0036445
3: 0.0012812, 0.0031128, 0.0014323, 0.0033564, -0.0015358, 0.0011528
4: 1.0017209, 1.0088270, 1.0023072, 1.0097717, -0.0059583, 0.0044726
5: 0.0025762, 0.0039586, 0.0026902, 0.0041424, -0.0011591, 0.0008701
6: -0.0108945, -0.0090955, -0.0111337, -0.0092439, -0.0011323, 0.0015084
7: -0.0101931, -0.0099636, -0.0102236, -0.0099825, -0.0001444, 0.0001924
8: -0.0045699, -0.0033269, -0.0044673, -0.0031616, -0.0010422, 0.0007823
9: -0.0015156, 0.0047070, -0.0023430, 0.0041935, -0.0039165, 0.0052176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023573, upper bound: 0.0022659
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023573, upper bound: 0.0022659
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040546, -0.0037262, -0.0040192, -0.0036983, -0.0002797, 0.0002161
1: 0.0003798, 0.0021982, 0.0002255, 0.0020025, -0.0011967, 0.0015486
2: 0.0100551, 0.0141177, 0.0104924, 0.0144624, -0.0034598, 0.0026735
3: 0.0013851, 0.0030971, 0.0012399, 0.0029128, -0.0011266, 0.0014580
4: 1.0021241, 1.0087658, 1.0015606, 1.0080508, -0.0043709, 0.0056564
5: 0.0026546, 0.0039467, 0.0025450, 0.0038076, -0.0008503, 0.0011004
6: -0.0108790, -0.0091976, -0.0106980, -0.0090549, -0.0014320, 0.0011066
7: -0.0101911, -0.0099766, -0.0101680, -0.0099584, -0.0001827, 0.0001412
8: -0.0044993, -0.0033376, -0.0045979, -0.0034626, -0.0007645, 0.0009894
9: -0.0014622, 0.0043538, -0.0008361, 0.0048473, -0.0049532, 0.0038275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0025163
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0025163
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040469, -0.0036945, -0.0040146, -0.0036999, -0.0002772, 0.0002450
1: 0.0002046, 0.0021555, 0.0002346, 0.0019766, -0.0013565, 0.0015349
2: 0.0101504, 0.0145091, 0.0105503, 0.0144421, -0.0034292, 0.0030306
3: 0.0012202, 0.0030569, 0.0012484, 0.0028884, -0.0012771, 0.0014451
4: 1.0014842, 1.0086100, 1.0015936, 1.0079563, -0.0049546, 0.0056064
5: 0.0025301, 0.0039164, 0.0025514, 0.0037892, -0.0009639, 0.0010907
6: -0.0108396, -0.0090356, -0.0106741, -0.0090633, -0.0014193, 0.0012543
7: -0.0101860, -0.0099559, -0.0101649, -0.0099595, -0.0001811, 0.0001600
8: -0.0046113, -0.0033648, -0.0045921, -0.0034792, -0.0008666, 0.0009807
9: -0.0013257, 0.0049142, -0.0007533, 0.0048183, -0.0049094, 0.0043386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022764, upper bound: 0.0023342
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024616, upper bound: 0.0024694
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040546, -0.0037262, -0.0040611, -0.0037249, -0.0002163, 0.0002211
1: 0.0003798, 0.0021982, 0.0003725, 0.0022343, -0.0012243, 0.0011975
2: 0.0100551, 0.0141177, 0.0099746, 0.0141340, -0.0026753, 0.0027352
3: 0.0013851, 0.0030971, 0.0013782, 0.0031310, -0.0011526, 0.0011274
4: 1.0021241, 1.0087658, 1.0020974, 1.0088975, -0.0044717, 0.0043738
5: 0.0026546, 0.0039467, 0.0026494, 0.0039723, -0.0008699, 0.0008509
6: -0.0108790, -0.0091976, -0.0109124, -0.0091908, -0.0011073, 0.0011321
7: -0.0101911, -0.0099766, -0.0101953, -0.0099757, -0.0001412, 0.0001444
8: -0.0044993, -0.0033376, -0.0045040, -0.0033145, -0.0007822, 0.0007650
9: -0.0014622, 0.0043538, -0.0015775, 0.0043773, -0.0038300, 0.0039158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0024862
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0024862
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040469, -0.0036945, -0.0040556, -0.0037265, -0.0002137, 0.0002564
1: 0.0002046, 0.0021555, 0.0003813, 0.0022041, -0.0014198, 0.0011834
2: 0.0101504, 0.0145091, 0.0100420, 0.0141142, -0.0026439, 0.0031719
3: 0.0012202, 0.0030569, 0.0013866, 0.0031026, -0.0013366, 0.0011141
4: 1.0014842, 1.0086100, 1.0021297, 1.0087873, -0.0051857, 0.0043224
5: 0.0025301, 0.0039164, 0.0026557, 0.0039509, -0.0010088, 0.0008409
6: -0.0108396, -0.0090356, -0.0108844, -0.0091990, -0.0010943, 0.0013128
7: -0.0101860, -0.0099559, -0.0101918, -0.0099768, -0.0001396, 0.0001675
8: -0.0046113, -0.0033648, -0.0044983, -0.0033338, -0.0009071, 0.0007561
9: -0.0013257, 0.0049142, -0.0014809, 0.0043489, -0.0037850, 0.0045410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022753, upper bound: 0.0022683
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024616, upper bound: 0.0024365
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.34 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025933, upper bound: 0.0025933
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025933, upper bound: 0.0025932
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025145, upper bound: 0.0025159
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025612, upper bound: 0.0025612
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0024806, upper bound: 0.0025433
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0024806, upper bound: 0.0025433
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0022964, upper bound: 0.0023486
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0023971, upper bound: 0.0024582
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025433, upper bound: 0.0024806
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025433, upper bound: 0.0024806
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0023406, upper bound: 0.0023064
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0024582, upper bound: 0.0023970
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025403, upper bound: 0.0024775
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025403, upper bound: 0.0024775
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0023176, upper bound: 0.0023020
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0024461, upper bound: 0.0023934
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0023869, upper bound: 0.0022890
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0023869, upper bound: 0.0022890
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0023573, upper bound: 0.0022659
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0023573, upper bound: 0.0022659
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0025163
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0025163
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0022764, upper bound: 0.0023342
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0024616, upper bound: 0.0024694
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0024862
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0025324, upper bound: 0.0024862
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0022753, upper bound: 0.0022683
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -0.0024616, upper bound: 0.0024365

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040125, -0.0036996, -0.0040125, -0.0036996, -0.0002097, 0.0002097
1: 0.0002329, 0.0019650, 0.0002329, 0.0019650, -0.0011611, 0.0011611
2: 0.0105762, 0.0144458, 0.0105762, 0.0144458, -0.0025941, 0.0025941
3: 0.0012468, 0.0028775, 0.0012468, 0.0028775, -0.0010932, 0.0010932
4: 1.0015875, 1.0079141, 1.0015875, 1.0079141, -0.0042411, 0.0042411
5: 0.0025502, 0.0037810, 0.0025502, 0.0037810, -0.0008251, 0.0008251
6: -0.0106634, -0.0090617, -0.0106634, -0.0090617, -0.0010737, 0.0010737
7: -0.0101636, -0.0099593, -0.0101636, -0.0099593, -0.0001370, 0.0001370
8: -0.0045932, -0.0034866, -0.0045932, -0.0034866, -0.0007418, 0.0007418
9: -0.0007162, 0.0048237, -0.0007162, 0.0048237, -0.0037138, 0.0037138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025134, upper bound: 0.0024432
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025264, upper bound: 0.0024807
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040125, -0.0036996, -0.0040070, -0.0036651, -0.0002506, 0.0002043
1: 0.0002329, 0.0019650, 0.0000417, 0.0019349, -0.0011312, 0.0013877
2: 0.0105762, 0.0144458, 0.0106433, 0.0148730, -0.0031004, 0.0025272
3: 0.0012468, 0.0028775, 0.0010669, 0.0028492, -0.0010649, 0.0013065
4: 1.0015875, 1.0079141, 1.0008893, 1.0078043, -0.0041316, 0.0050687
5: 0.0025502, 0.0037810, 0.0024144, 0.0037596, -0.0008038, 0.0009861
6: -0.0106634, -0.0090617, -0.0106356, -0.0088849, -0.0012832, 0.0010460
7: -0.0101636, -0.0099593, -0.0101600, -0.0099367, -0.0001637, 0.0001334
8: -0.0045932, -0.0034866, -0.0047153, -0.0035058, -0.0007227, 0.0008866
9: -0.0007162, 0.0048237, -0.0006201, 0.0054351, -0.0044385, 0.0036179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025134, upper bound: 0.0024432
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025264, upper bound: 0.0024807
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036717, -0.0040282, -0.0037139, -0.0001871, 0.0002463
1: 0.0000780, 0.0019345, 0.0003117, 0.0020519, -0.0013640, 0.0010357
2: 0.0106442, 0.0147919, 0.0103821, 0.0142697, -0.0023139, 0.0030473
3: 0.0011010, 0.0028489, 0.0013210, 0.0029593, -0.0012841, 0.0009751
4: 1.0010217, 1.0078028, 1.0018755, 1.0082313, -0.0049820, 0.0037829
5: 0.0024402, 0.0037594, 0.0026062, 0.0038427, -0.0009692, 0.0007359
6: -0.0106352, -0.0089185, -0.0107437, -0.0091346, -0.0009577, 0.0012613
7: -0.0101600, -0.0099410, -0.0101738, -0.0099686, -0.0001222, 0.0001609
8: -0.0046921, -0.0035060, -0.0045428, -0.0034311, -0.0008714, 0.0006617
9: -0.0006189, 0.0053191, -0.0009941, 0.0045716, -0.0033126, 0.0043626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025102, upper bound: 0.0025102
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025102, upper bound: 0.0025143
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036657, -0.0040145, -0.0037066, -0.0001887, 0.0002511
1: 0.0000450, 0.0019349, 0.0002712, 0.0019762, -0.0013901, 0.0010449
2: 0.0106434, 0.0148656, 0.0105511, 0.0143603, -0.0023343, 0.0031056
3: 0.0010700, 0.0028492, 0.0012829, 0.0028881, -0.0013087, 0.0009837
4: 1.0009013, 1.0078042, 1.0017275, 1.0079550, -0.0050772, 0.0038164
5: 0.0024167, 0.0037596, 0.0025774, 0.0037890, -0.0009877, 0.0007424
6: -0.0106355, -0.0088880, -0.0106737, -0.0090971, -0.0009662, 0.0012854
7: -0.0101600, -0.0099371, -0.0101649, -0.0099638, -0.0001232, 0.0001640
8: -0.0047132, -0.0035058, -0.0045687, -0.0034794, -0.0008881, 0.0006675
9: -0.0006200, 0.0054246, -0.0007522, 0.0047012, -0.0033419, 0.0044460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025143, upper bound: 0.0025145
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025143, upper bound: 0.0025612
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040125, -0.0036996, -0.0040576, -0.0037062, -0.0002116, 0.0002648
1: 0.0002329, 0.0019650, 0.0002694, 0.0022149, -0.0014660, 0.0011716
2: 0.0105762, 0.0144458, 0.0100178, 0.0143643, -0.0026175, 0.0032752
3: 0.0012468, 0.0028775, 0.0012812, 0.0031128, -0.0013802, 0.0011030
4: 1.0015875, 1.0079141, 1.0017209, 1.0088270, -0.0053546, 0.0042793
5: 0.0025502, 0.0037810, 0.0025762, 0.0039586, -0.0010417, 0.0008325
6: -0.0106634, -0.0090617, -0.0108945, -0.0090955, -0.0010834, 0.0013556
7: -0.0101636, -0.0099593, -0.0101931, -0.0099636, -0.0001382, 0.0001729
8: -0.0045932, -0.0034866, -0.0045699, -0.0033269, -0.0009366, 0.0007485
9: -0.0007162, 0.0048237, -0.0015156, 0.0047070, -0.0037473, 0.0046888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023801, upper bound: 0.0023298
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024439, upper bound: 0.0024332
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040125, -0.0036996, -0.0040507, -0.0036735, -0.0002515, 0.0002577
1: 0.0002329, 0.0019650, 0.0000881, 0.0021765, -0.0014267, 0.0013924
2: 0.0105762, 0.0144458, 0.0101037, 0.0147694, -0.0031107, 0.0031874
3: 0.0012468, 0.0028775, 0.0011105, 0.0030766, -0.0013432, 0.0013109
4: 1.0015875, 1.0079141, 1.0010585, 1.0086864, -0.0052109, 0.0050857
5: 0.0025502, 0.0037810, 0.0024473, 0.0039313, -0.0010137, 0.0009894
6: -0.0106634, -0.0090617, -0.0108589, -0.0089278, -0.0012875, 0.0013192
7: -0.0101636, -0.0099593, -0.0101885, -0.0099422, -0.0001642, 0.0001683
8: -0.0045932, -0.0034866, -0.0046857, -0.0033515, -0.0009115, 0.0008896
9: -0.0007162, 0.0048237, -0.0013927, 0.0052869, -0.0044534, 0.0045631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023801, upper bound: 0.0023298
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024439, upper bound: 0.0024332
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036657, -0.0040592, -0.0037131, -0.0001933, 0.0003059
1: 0.0000450, 0.0019349, 0.0003076, 0.0022239, -0.0016936, 0.0010703
2: 0.0106434, 0.0148656, 0.0099977, 0.0142790, -0.0023913, 0.0037837
3: 0.0010700, 0.0028492, 0.0013171, 0.0031213, -0.0015945, 0.0010077
4: 1.0009013, 1.0078042, 1.0018603, 1.0088596, -0.0061859, 0.0039094
5: 0.0024167, 0.0037596, 0.0026033, 0.0039650, -0.0012034, 0.0007605
6: -0.0106355, -0.0088880, -0.0109028, -0.0091308, -0.0009897, 0.0015661
7: -0.0101600, -0.0099371, -0.0101941, -0.0099681, -0.0001262, 0.0001998
8: -0.0047132, -0.0035058, -0.0045455, -0.0033212, -0.0010820, 0.0006838
9: -0.0006200, 0.0054246, -0.0015443, 0.0045848, -0.0034234, 0.0054169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023064, upper bound: 0.0023406
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0023064, upper bound: 0.0024582
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0040125, -0.0036996, -0.0002648, 0.0002116
1: 0.0002694, 0.0022149, 0.0002329, 0.0019650, -0.0011716, 0.0014660
2: 0.0100178, 0.0143643, 0.0105762, 0.0144458, -0.0032752, 0.0026175
3: 0.0012812, 0.0031128, 0.0012468, 0.0028775, -0.0011030, 0.0013802
4: 1.0017209, 1.0088270, 1.0015875, 1.0079141, -0.0042793, 0.0053546
5: 0.0025762, 0.0039586, 0.0025502, 0.0037810, -0.0008325, 0.0010417
6: -0.0108945, -0.0090955, -0.0106634, -0.0090617, -0.0013556, 0.0010834
7: -0.0101931, -0.0099636, -0.0101636, -0.0099593, -0.0001729, 0.0001382
8: -0.0045699, -0.0033269, -0.0045932, -0.0034866, -0.0007485, 0.0009366
9: -0.0015156, 0.0047070, -0.0007162, 0.0048237, -0.0046888, 0.0037473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019206, upper bound: 0.0016994
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025814, upper bound: 0.0024098
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0040070, -0.0036651, -0.0003057, 0.0002062
1: 0.0002694, 0.0022149, 0.0000417, 0.0019349, -0.0011416, 0.0016926
2: 0.0100178, 0.0143643, 0.0106433, 0.0148730, -0.0037814, 0.0025505
3: 0.0012812, 0.0031128, 0.0010669, 0.0028492, -0.0010748, 0.0015935
4: 1.0017209, 1.0088270, 1.0008893, 1.0078043, -0.0041698, 0.0061822
5: 0.0025762, 0.0039586, 0.0024144, 0.0037596, -0.0008112, 0.0012027
6: -0.0108945, -0.0090955, -0.0106356, -0.0088849, -0.0015651, 0.0010556
7: -0.0101931, -0.0099636, -0.0101600, -0.0099367, -0.0001996, 0.0001347
8: -0.0045699, -0.0033269, -0.0047153, -0.0035058, -0.0007294, 0.0010814
9: -0.0015156, 0.0047070, -0.0006201, 0.0054351, -0.0054136, 0.0036514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019206, upper bound: 0.0016994
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025814, upper bound: 0.0024098
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040507, -0.0036741, -0.0040145, -0.0037066, -0.0002454, 0.0002518
1: 0.0000915, 0.0021764, 0.0002712, 0.0019762, -0.0013944, 0.0013588
2: 0.0101037, 0.0147616, 0.0105511, 0.0143603, -0.0030357, 0.0031152
3: 0.0011138, 0.0030766, 0.0012829, 0.0028881, -0.0013128, 0.0012793
4: 1.0010713, 1.0086863, 1.0017275, 1.0079550, -0.0050930, 0.0049630
5: 0.0024498, 0.0039312, 0.0025774, 0.0037890, -0.0009908, 0.0009655
6: -0.0108589, -0.0089310, -0.0106737, -0.0090971, -0.0012565, 0.0012894
7: -0.0101885, -0.0099426, -0.0101649, -0.0099638, -0.0001603, 0.0001645
8: -0.0046835, -0.0033515, -0.0045687, -0.0034794, -0.0008909, 0.0008681
9: -0.0013926, 0.0052758, -0.0007522, 0.0047012, -0.0043460, 0.0044598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023486, upper bound: 0.0022965
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023486, upper bound: 0.0023971
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0040576, -0.0037062, -0.0002202, 0.0002202
1: 0.0002694, 0.0022149, 0.0002694, 0.0022149, -0.0012193, 0.0012193
2: 0.0100178, 0.0143643, 0.0100178, 0.0143643, -0.0027241, 0.0027241
3: 0.0012812, 0.0031128, 0.0012812, 0.0031128, -0.0011479, 0.0011479
4: 1.0017209, 1.0088270, 1.0017209, 1.0088270, -0.0044536, 0.0044536
5: 0.0025762, 0.0039586, 0.0025762, 0.0039586, -0.0008664, 0.0008664
6: -0.0108945, -0.0090955, -0.0108945, -0.0090955, -0.0011275, 0.0011275
7: -0.0101931, -0.0099636, -0.0101931, -0.0099636, -0.0001438, 0.0001438
8: -0.0045699, -0.0033269, -0.0045699, -0.0033269, -0.0007790, 0.0007790
9: -0.0015156, 0.0047070, -0.0015156, 0.0047070, -0.0038999, 0.0038999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024202, upper bound: 0.0022939
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025356, upper bound: 0.0023688
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040576, -0.0037062, -0.0040507, -0.0036735, -0.0002614, 0.0002149
1: 0.0002694, 0.0022149, 0.0000881, 0.0021765, -0.0011897, 0.0014473
2: 0.0100178, 0.0143643, 0.0101037, 0.0147694, -0.0032335, 0.0026580
3: 0.0012812, 0.0031128, 0.0011105, 0.0030766, -0.0011201, 0.0013626
4: 1.0017209, 1.0088270, 1.0010585, 1.0086864, -0.0043456, 0.0052864
5: 0.0025762, 0.0039586, 0.0024473, 0.0039313, -0.0008454, 0.0010284
6: -0.0108945, -0.0090955, -0.0108589, -0.0089278, -0.0013383, 0.0011001
7: -0.0101931, -0.0099636, -0.0101885, -0.0099422, -0.0001707, 0.0001403
8: -0.0045699, -0.0033269, -0.0046857, -0.0033515, -0.0007601, 0.0009247
9: -0.0015156, 0.0047070, -0.0013927, 0.0052869, -0.0046292, 0.0038053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024202, upper bound: 0.0022939
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025356, upper bound: 0.0023688
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040507, -0.0036741, -0.0040592, -0.0037131, -0.0002000, 0.0002619
1: 0.0000915, 0.0021764, 0.0003076, 0.0022239, -0.0014502, 0.0011076
2: 0.0101037, 0.0147616, 0.0099977, 0.0142790, -0.0024746, 0.0032399
3: 0.0011138, 0.0030766, 0.0013171, 0.0031213, -0.0013653, 0.0010428
4: 1.0010713, 1.0086863, 1.0018603, 1.0088596, -0.0052968, 0.0040456
5: 0.0024498, 0.0039312, 0.0026033, 0.0039650, -0.0010304, 0.0007870
6: -0.0108589, -0.0089310, -0.0109028, -0.0091308, -0.0010242, 0.0013410
7: -0.0101885, -0.0099426, -0.0101941, -0.0099681, -0.0001306, 0.0001711
8: -0.0046835, -0.0033515, -0.0045455, -0.0033212, -0.0009265, 0.0007076
9: -0.0013926, 0.0052758, -0.0015443, 0.0045848, -0.0035426, 0.0046382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023267, upper bound: 0.0022912
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023267, upper bound: 0.0023934
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040546, -0.0037262, -0.0040125, -0.0036996, -0.0002787, 0.0002101
1: 0.0003798, 0.0021982, 0.0002329, 0.0019650, -0.0011634, 0.0015431
2: 0.0100551, 0.0141177, 0.0105762, 0.0144458, -0.0034474, 0.0025992
3: 0.0013851, 0.0030971, 0.0012468, 0.0028775, -0.0010953, 0.0014528
4: 1.0021241, 1.0087658, 1.0015875, 1.0079141, -0.0042494, 0.0056361
5: 0.0026546, 0.0039467, 0.0025502, 0.0037810, -0.0008267, 0.0010965
6: -0.0108790, -0.0091976, -0.0106634, -0.0090617, -0.0014269, 0.0010758
7: -0.0101911, -0.0099766, -0.0101636, -0.0099593, -0.0001820, 0.0001372
8: -0.0044993, -0.0033376, -0.0045932, -0.0034866, -0.0007433, 0.0009859
9: -0.0014622, 0.0043538, -0.0007162, 0.0048237, -0.0049354, 0.0037211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022970
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0024028
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040546, -0.0037262, -0.0040070, -0.0036651, -0.0003196, 0.0002047
1: 0.0003798, 0.0021982, 0.0000417, 0.0019349, -0.0011334, 0.0017697
2: 0.0100551, 0.0141177, 0.0106433, 0.0148730, -0.0039537, 0.0025322
3: 0.0013851, 0.0030971, 0.0010669, 0.0028492, -0.0010671, 0.0016661
4: 1.0021241, 1.0087658, 1.0008893, 1.0078043, -0.0041399, 0.0064638
5: 0.0026546, 0.0039467, 0.0024144, 0.0037596, -0.0008054, 0.0012575
6: -0.0108790, -0.0091976, -0.0106356, -0.0088849, -0.0016364, 0.0010481
7: -0.0101911, -0.0099766, -0.0101600, -0.0099367, -0.0002087, 0.0001337
8: -0.0044993, -0.0033376, -0.0047153, -0.0035058, -0.0007241, 0.0011306
9: -0.0014622, 0.0043538, -0.0006201, 0.0054351, -0.0056602, 0.0036252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022970
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0024028
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040469, -0.0036951, -0.0040145, -0.0037066, -0.0002568, 0.0002446
1: 0.0002078, 0.0021555, 0.0002712, 0.0019762, -0.0013541, 0.0014221
2: 0.0101505, 0.0145019, 0.0105511, 0.0143603, -0.0031772, 0.0030253
3: 0.0012232, 0.0030569, 0.0012829, 0.0028881, -0.0012749, 0.0013389
4: 1.0014960, 1.0086099, 1.0017275, 1.0079550, -0.0049460, 0.0051944
5: 0.0025324, 0.0039164, 0.0025774, 0.0037890, -0.0009622, 0.0010105
6: -0.0108395, -0.0090385, -0.0106737, -0.0090971, -0.0013150, 0.0012522
7: -0.0101860, -0.0099563, -0.0101649, -0.0099638, -0.0001677, 0.0001597
8: -0.0046092, -0.0033649, -0.0045687, -0.0034794, -0.0008651, 0.0009086
9: -0.0013256, 0.0049039, -0.0007522, 0.0047012, -0.0045486, 0.0043311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022691, upper bound: 0.0023169
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022691, upper bound: 0.0024694
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040546, -0.0037262, -0.0040546, -0.0037262, -0.0002153, 0.0002153
1: 0.0003798, 0.0021982, 0.0003798, 0.0021982, -0.0011921, 0.0011921
2: 0.0100551, 0.0141177, 0.0100551, 0.0141177, -0.0026632, 0.0026632
3: 0.0013851, 0.0030971, 0.0013851, 0.0030971, -0.0011223, 0.0011223
4: 1.0021241, 1.0087658, 1.0021241, 1.0087658, -0.0043540, 0.0043540
5: 0.0026546, 0.0039467, 0.0026546, 0.0039467, -0.0008470, 0.0008470
6: -0.0108790, -0.0091976, -0.0108790, -0.0091976, -0.0011023, 0.0011023
7: -0.0101911, -0.0099766, -0.0101911, -0.0099766, -0.0001406, 0.0001406
8: -0.0044993, -0.0033376, -0.0044993, -0.0033376, -0.0007616, 0.0007616
9: -0.0014622, 0.0043538, -0.0014622, 0.0043538, -0.0038127, 0.0038127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022523
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0023778
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040546, -0.0037262, -0.0040469, -0.0036945, -0.0002559, 0.0002093
1: 0.0003798, 0.0021982, 0.0002046, 0.0021555, -0.0011591, 0.0014169
2: 0.0100551, 0.0141177, 0.0101504, 0.0145091, -0.0031655, 0.0025895
3: 0.0013851, 0.0030971, 0.0012202, 0.0030569, -0.0010912, 0.0013340
4: 1.0021241, 1.0087658, 1.0014842, 1.0086100, -0.0042335, 0.0051753
5: 0.0026546, 0.0039467, 0.0025301, 0.0039164, -0.0008236, 0.0010068
6: -0.0108790, -0.0091976, -0.0108396, -0.0090356, -0.0013102, 0.0010718
7: -0.0101911, -0.0099766, -0.0101860, -0.0099559, -0.0001671, 0.0001367
8: -0.0044993, -0.0033376, -0.0046113, -0.0033648, -0.0007405, 0.0009052
9: -0.0014622, 0.0043538, -0.0013257, 0.0049142, -0.0045319, 0.0037072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022523
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0023778
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040469, -0.0036951, -0.0040556, -0.0037325, -0.0001947, 0.0002560
1: 0.0002078, 0.0021555, 0.0004145, 0.0022037, -0.0014173, 0.0010782
2: 0.0101505, 0.0145019, 0.0100429, 0.0140400, -0.0024088, 0.0031663
3: 0.0012232, 0.0030569, 0.0014178, 0.0031023, -0.0013343, 0.0010151
4: 1.0014960, 1.0086099, 1.0022510, 1.0087858, -0.0051766, 0.0039381
5: 0.0025324, 0.0039164, 0.0026793, 0.0039506, -0.0010071, 0.0007661
6: -0.0108395, -0.0090385, -0.0108841, -0.0092297, -0.0009970, 0.0013105
7: -0.0101860, -0.0099563, -0.0101917, -0.0099807, -0.0001272, 0.0001672
8: -0.0046092, -0.0033649, -0.0044771, -0.0033341, -0.0009055, 0.0006888
9: -0.0013256, 0.0049039, -0.0014797, 0.0042427, -0.0034485, 0.0045330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022690, upper bound: 0.0022587
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022690, upper bound: 0.0024365
time: 0.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.43 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025134, upper bound: 0.0024432
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025264, upper bound: 0.0024807
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025134, upper bound: 0.0024432
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025264, upper bound: 0.0024807
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025102, upper bound: 0.0025102
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025102, upper bound: 0.0025143
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025143, upper bound: 0.0025145
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025143, upper bound: 0.0025612
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023801, upper bound: 0.0023298
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024439, upper bound: 0.0024332
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023801, upper bound: 0.0023298
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024439, upper bound: 0.0024332
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023064, upper bound: 0.0023406
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023064, upper bound: 0.0024582
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0019206, upper bound: 0.0016994
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025814, upper bound: 0.0024098
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0019206, upper bound: 0.0016994
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025814, upper bound: 0.0024098
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023486, upper bound: 0.0022965
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023486, upper bound: 0.0023971
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024202, upper bound: 0.0022939
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025356, upper bound: 0.0023688
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024202, upper bound: 0.0022939
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0025356, upper bound: 0.0023688
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023267, upper bound: 0.0022912
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023267, upper bound: 0.0023934
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022970
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0024028
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022970
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0024028
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0022691, upper bound: 0.0023169
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0022691, upper bound: 0.0024694
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022523
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0023778
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0023283, upper bound: 0.0022523
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0024733, upper bound: 0.0023778
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0022690, upper bound: 0.0022587
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0022690, upper bound: 0.0024365

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040266, -0.0037136, -0.0040124, -0.0037058, -0.0002049, 0.0001879
1: 0.0003099, 0.0020432, 0.0002672, 0.0019645, -0.0010406, 0.0011344
2: 0.0104015, 0.0142738, 0.0105772, 0.0143691, -0.0025345, 0.0023248
3: 0.0013193, 0.0029511, 0.0012792, 0.0028771, -0.0009797, 0.0010680
4: 1.0018688, 1.0081997, 1.0017130, 1.0079123, -0.0038008, 0.0041435
5: 0.0026050, 0.0038366, 0.0025746, 0.0037807, -0.0007394, 0.0008061
6: -0.0107357, -0.0091329, -0.0106629, -0.0090935, -0.0010490, 0.0009622
7: -0.0101728, -0.0099683, -0.0101635, -0.0099633, -0.0001338, 0.0001227
8: -0.0045440, -0.0034366, -0.0045712, -0.0034869, -0.0006648, 0.0007248
9: -0.0009664, 0.0045774, -0.0007147, 0.0047138, -0.0036284, 0.0033283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024239, upper bound: 0.0024925
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025404, upper bound: 0.0025322
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040124, -0.0037063, -0.0040125, -0.0037003, -0.0002093, 0.0001901
1: 0.0002695, 0.0019646, 0.0002366, 0.0019649, -0.0010526, 0.0011589
2: 0.0105771, 0.0143641, 0.0105763, 0.0144376, -0.0025892, 0.0023517
3: 0.0012813, 0.0028772, 0.0012503, 0.0028775, -0.0009910, 0.0010911
4: 1.0017211, 1.0079125, 1.0016011, 1.0079139, -0.0038448, 0.0042330
5: 0.0025762, 0.0037807, 0.0025529, 0.0037810, -0.0007480, 0.0008235
6: -0.0106630, -0.0090956, -0.0106633, -0.0090652, -0.0010716, 0.0009734
7: -0.0101635, -0.0099636, -0.0101636, -0.0099597, -0.0001367, 0.0001242
8: -0.0045698, -0.0034868, -0.0045908, -0.0034866, -0.0006725, 0.0007404
9: -0.0007150, 0.0047066, -0.0007161, 0.0048118, -0.0037067, 0.0033668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025036, upper bound: 0.0025505
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025563, upper bound: 0.0025563
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040266, -0.0037136, -0.0040070, -0.0036717, -0.0002468, 0.0001825
1: 0.0003099, 0.0020432, 0.0000780, 0.0019345, -0.0010107, 0.0013668
2: 0.0104015, 0.0142738, 0.0106442, 0.0147919, -0.0030535, 0.0022580
3: 0.0013193, 0.0029511, 0.0011010, 0.0028489, -0.0009515, 0.0012868
4: 1.0018688, 1.0081997, 1.0010217, 1.0078028, -0.0036916, 0.0049921
5: 0.0026050, 0.0038366, 0.0024402, 0.0037594, -0.0007182, 0.0009712
6: -0.0107357, -0.0091329, -0.0106352, -0.0089185, -0.0012638, 0.0009346
7: -0.0101728, -0.0099683, -0.0101600, -0.0099410, -0.0001612, 0.0001192
8: -0.0045440, -0.0034366, -0.0046921, -0.0035060, -0.0006457, 0.0008732
9: -0.0009664, 0.0045774, -0.0006189, 0.0053191, -0.0043715, 0.0032326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025164, upper bound: 0.0024431
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025164, upper bound: 0.0024432
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040124, -0.0037063, -0.0040070, -0.0036657, -0.0002502, 0.0001838
1: 0.0002695, 0.0019646, 0.0000450, 0.0019349, -0.0010179, 0.0013853
2: 0.0105771, 0.0143641, 0.0106434, 0.0148656, -0.0030948, 0.0022741
3: 0.0012813, 0.0028772, 0.0010700, 0.0028492, -0.0009583, 0.0013042
4: 1.0017211, 1.0079125, 1.0009013, 1.0078042, -0.0037179, 0.0050597
5: 0.0025762, 0.0037807, 0.0024167, 0.0037596, -0.0007233, 0.0009843
6: -0.0106630, -0.0090956, -0.0106355, -0.0088880, -0.0012809, 0.0009412
7: -0.0101635, -0.0099636, -0.0101600, -0.0099371, -0.0001634, 0.0001201
8: -0.0045698, -0.0034868, -0.0047132, -0.0035058, -0.0006503, 0.0008850
9: -0.0007150, 0.0047066, -0.0006200, 0.0054246, -0.0044306, 0.0032557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025195, upper bound: 0.0024477
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025195, upper bound: 0.0024807
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040202, -0.0036791, -0.0040282, -0.0037139, -0.0001911, 0.0002364
1: 0.0001189, 0.0020078, 0.0003117, 0.0020519, -0.0013092, 0.0010584
2: 0.0104806, 0.0147006, 0.0103821, 0.0142697, -0.0023645, 0.0029249
3: 0.0011395, 0.0029178, 0.0013210, 0.0029593, -0.0012326, 0.0009964
4: 1.0011711, 1.0080702, 1.0018755, 1.0082313, -0.0047819, 0.0038657
5: 0.0024692, 0.0038114, 0.0026062, 0.0038427, -0.0009303, 0.0007520
6: -0.0107029, -0.0089563, -0.0107437, -0.0091346, -0.0009787, 0.0012106
7: -0.0101686, -0.0099458, -0.0101738, -0.0099686, -0.0001248, 0.0001544
8: -0.0046660, -0.0034593, -0.0045428, -0.0034311, -0.0008364, 0.0006762
9: -0.0008531, 0.0051884, -0.0009941, 0.0045716, -0.0033851, 0.0041874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025152
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025152
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040282, -0.0037139, -0.0001871, 0.0002516
1: 0.0000745, 0.0019346, 0.0003117, 0.0020519, -0.0013931, 0.0010358
2: 0.0106440, 0.0147997, 0.0103821, 0.0142697, -0.0023141, 0.0031124
3: 0.0010977, 0.0028490, 0.0013210, 0.0029593, -0.0013116, 0.0009752
4: 1.0010091, 1.0078032, 1.0018755, 1.0082313, -0.0050883, 0.0037832
5: 0.0024377, 0.0037594, 0.0026062, 0.0038427, -0.0009899, 0.0007360
6: -0.0106353, -0.0089153, -0.0107437, -0.0091346, -0.0009578, 0.0012882
7: -0.0101600, -0.0099406, -0.0101738, -0.0099686, -0.0001222, 0.0001643
8: -0.0046944, -0.0035060, -0.0045428, -0.0034311, -0.0008900, 0.0006618
9: -0.0006192, 0.0053303, -0.0009941, 0.0045716, -0.0033129, 0.0044557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025159
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025159
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040202, -0.0036791, -0.0040145, -0.0037066, -0.0002099, 0.0002328
1: 0.0001189, 0.0020078, 0.0002712, 0.0019762, -0.0012891, 0.0011623
2: 0.0104806, 0.0147006, 0.0105511, 0.0143603, -0.0025968, 0.0028800
3: 0.0011395, 0.0029178, 0.0012829, 0.0028881, -0.0012136, 0.0010943
4: 1.0011711, 1.0080702, 1.0017275, 1.0079550, -0.0047084, 0.0042454
5: 0.0024692, 0.0038114, 0.0025774, 0.0037890, -0.0009160, 0.0008259
6: -0.0107029, -0.0089563, -0.0106737, -0.0090971, -0.0010748, 0.0011920
7: -0.0101686, -0.0099458, -0.0101649, -0.0099638, -0.0001371, 0.0001521
8: -0.0046660, -0.0034593, -0.0045687, -0.0034794, -0.0008236, 0.0007426
9: -0.0008531, 0.0051884, -0.0007522, 0.0047012, -0.0037176, 0.0041230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025145
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025145
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040145, -0.0037066, -0.0001887, 0.0002323
1: 0.0000745, 0.0019346, 0.0002712, 0.0019762, -0.0012865, 0.0010446
2: 0.0106440, 0.0147997, 0.0105511, 0.0143603, -0.0023338, 0.0028742
3: 0.0010977, 0.0028490, 0.0012829, 0.0028881, -0.0012112, 0.0009835
4: 1.0010091, 1.0078032, 1.0017275, 1.0079550, -0.0046989, 0.0038154
5: 0.0024377, 0.0037594, 0.0025774, 0.0037890, -0.0009141, 0.0007423
6: -0.0106353, -0.0089153, -0.0106737, -0.0090971, -0.0009659, 0.0011896
7: -0.0101600, -0.0099406, -0.0101649, -0.0099638, -0.0001232, 0.0001517
8: -0.0046944, -0.0035060, -0.0045687, -0.0034794, -0.0008219, 0.0006674
9: -0.0006192, 0.0053303, -0.0007522, 0.0047012, -0.0033411, 0.0041147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025611
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025611
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040124, -0.0037063, -0.0040576, -0.0037069, -0.0002111, 0.0002466
1: 0.0002695, 0.0019646, 0.0002730, 0.0022149, -0.0013656, 0.0011687
2: 0.0105771, 0.0143641, 0.0100179, 0.0143561, -0.0026111, 0.0030509
3: 0.0012813, 0.0028772, 0.0012846, 0.0031128, -0.0012856, 0.0011003
4: 1.0017211, 1.0079125, 1.0017343, 1.0088269, -0.0049878, 0.0042689
5: 0.0025762, 0.0037807, 0.0025788, 0.0039586, -0.0009703, 0.0008305
6: -0.0106630, -0.0090956, -0.0108944, -0.0090989, -0.0010807, 0.0012627
7: -0.0101635, -0.0099636, -0.0101930, -0.0099640, -0.0001379, 0.0001611
8: -0.0045698, -0.0034868, -0.0045675, -0.0033269, -0.0008724, 0.0007467
9: -0.0007150, 0.0047066, -0.0015155, 0.0046952, -0.0037381, 0.0043677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024469, upper bound: 0.0026135
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025075, upper bound: 0.0026479
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040124, -0.0037063, -0.0040507, -0.0036741, -0.0002510, 0.0002382
1: 0.0002695, 0.0019646, 0.0000915, 0.0021764, -0.0013190, 0.0013896
2: 0.0105771, 0.0143641, 0.0101037, 0.0147616, -0.0031045, 0.0029468
3: 0.0012813, 0.0028772, 0.0011138, 0.0030766, -0.0012418, 0.0013082
4: 1.0017211, 1.0079125, 1.0010713, 1.0086863, -0.0048177, 0.0050755
5: 0.0025762, 0.0037807, 0.0024498, 0.0039312, -0.0009372, 0.0009874
6: -0.0106630, -0.0090956, -0.0108589, -0.0089310, -0.0012849, 0.0012197
7: -0.0101635, -0.0099636, -0.0101885, -0.0099426, -0.0001639, 0.0001556
8: -0.0045698, -0.0034868, -0.0046835, -0.0033515, -0.0008427, 0.0008878
9: -0.0007150, 0.0047066, -0.0013926, 0.0052758, -0.0044445, 0.0042187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023797, upper bound: 0.0023449
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0023797, upper bound: 0.0024332
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040592, -0.0037131, -0.0001933, 0.0002888
1: 0.0000745, 0.0019346, 0.0003076, 0.0022239, -0.0015988, 0.0010701
2: 0.0106440, 0.0147997, 0.0099977, 0.0142790, -0.0023907, 0.0035720
3: 0.0010977, 0.0028490, 0.0013171, 0.0031213, -0.0015052, 0.0010074
4: 1.0010091, 1.0078032, 1.0018603, 1.0088596, -0.0058398, 0.0039085
5: 0.0024377, 0.0037594, 0.0026033, 0.0039650, -0.0011361, 0.0007604
6: -0.0106353, -0.0089153, -0.0109028, -0.0091308, -0.0009895, 0.0014784
7: -0.0101600, -0.0099406, -0.0101941, -0.0099681, -0.0001262, 0.0001886
8: -0.0046944, -0.0035060, -0.0045455, -0.0033212, -0.0010215, 0.0006837
9: -0.0006192, 0.0053303, -0.0015443, 0.0045848, -0.0034226, 0.0051138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022774, upper bound: 0.0024581
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022774, upper bound: 0.0024580
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040544, -0.0037067, -0.0040124, -0.0036997, -0.0002458, 0.0002110
1: 0.0002720, 0.0021970, 0.0002329, 0.0019645, -0.0011682, 0.0013613
2: 0.0100577, 0.0143584, 0.0105773, 0.0144457, -0.0030412, 0.0026098
3: 0.0012837, 0.0030960, 0.0012469, 0.0028771, -0.0010998, 0.0012816
4: 1.0017306, 1.0087616, 1.0015877, 1.0079123, -0.0042668, 0.0049720
5: 0.0025781, 0.0039459, 0.0025503, 0.0037806, -0.0008301, 0.0009672
6: -0.0108779, -0.0090979, -0.0106629, -0.0090618, -0.0012587, 0.0010802
7: -0.0101909, -0.0099639, -0.0101635, -0.0099593, -0.0001606, 0.0001378
8: -0.0045682, -0.0033383, -0.0045931, -0.0034869, -0.0007463, 0.0008697
9: -0.0014585, 0.0046985, -0.0007147, 0.0048235, -0.0043538, 0.0037363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025713, upper bound: 0.0024656
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026479, upper bound: 0.0025075
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040544, -0.0037067, -0.0040070, -0.0036651, -0.0002835, 0.0002056
1: 0.0002720, 0.0021970, 0.0000418, 0.0019345, -0.0011383, 0.0015697
2: 0.0100577, 0.0143584, 0.0106442, 0.0148728, -0.0035070, 0.0025431
3: 0.0012837, 0.0030960, 0.0010669, 0.0028489, -0.0010717, 0.0014779
4: 1.0017306, 1.0087616, 1.0008894, 1.0078027, -0.0041577, 0.0057335
5: 0.0025781, 0.0039459, 0.0024144, 0.0037594, -0.0008088, 0.0011154
6: -0.0108779, -0.0090979, -0.0106352, -0.0088850, -0.0014515, 0.0010526
7: -0.0101909, -0.0099639, -0.0101600, -0.0099367, -0.0001852, 0.0001343
8: -0.0045682, -0.0033383, -0.0047153, -0.0035060, -0.0007272, 0.0010029
9: -0.0014585, 0.0046985, -0.0006189, 0.0054349, -0.0050207, 0.0036408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023820, upper bound: 0.0022403
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024766, upper bound: 0.0023005
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040575, -0.0037128, -0.0040576, -0.0037069, -0.0002198, 0.0002010
1: 0.0003056, 0.0022145, 0.0002730, 0.0022149, -0.0011130, 0.0012172
2: 0.0100186, 0.0142833, 0.0100179, 0.0143561, -0.0027194, 0.0024865
3: 0.0013153, 0.0031125, 0.0012846, 0.0031128, -0.0010478, 0.0011460
4: 1.0018532, 1.0088255, 1.0017343, 1.0088269, -0.0040652, 0.0044459
5: 0.0026019, 0.0039583, 0.0025788, 0.0039586, -0.0007908, 0.0008649
6: -0.0108941, -0.0091290, -0.0108944, -0.0090989, -0.0011255, 0.0010292
7: -0.0101930, -0.0099678, -0.0101930, -0.0099640, -0.0001436, 0.0001313
8: -0.0045467, -0.0033271, -0.0045675, -0.0033269, -0.0007111, 0.0007777
9: -0.0015144, 0.0045910, -0.0015155, 0.0046952, -0.0038931, 0.0035598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025469, upper bound: 0.0024925
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026479, upper bound: 0.0025075
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040575, -0.0037128, -0.0040507, -0.0036741, -0.0002610, 0.0001945
1: 0.0003056, 0.0022145, 0.0000915, 0.0021764, -0.0010771, 0.0014449
2: 0.0100186, 0.0142833, 0.0101037, 0.0147616, -0.0032281, 0.0024065
3: 0.0013153, 0.0031125, 0.0011138, 0.0030766, -0.0010141, 0.0013603
4: 1.0018532, 1.0088255, 1.0010713, 1.0086863, -0.0039343, 0.0052775
5: 0.0026019, 0.0039583, 0.0024498, 0.0039312, -0.0007654, 0.0010267
6: -0.0108941, -0.0091290, -0.0108589, -0.0089310, -0.0013361, 0.0009960
7: -0.0101930, -0.0099678, -0.0101885, -0.0099426, -0.0001704, 0.0001271
8: -0.0045467, -0.0033271, -0.0046835, -0.0033515, -0.0006882, 0.0009231
9: -0.0015144, 0.0045910, -0.0013926, 0.0052758, -0.0046214, 0.0034452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024300, upper bound: 0.0023021
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024300, upper bound: 0.0023688
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040545, -0.0037321, -0.0040125, -0.0037003, -0.0002783, 0.0001897
1: 0.0004128, 0.0021978, 0.0002366, 0.0019649, -0.0010506, 0.0015409
2: 0.0100560, 0.0140438, 0.0105763, 0.0144376, -0.0034425, 0.0023472
3: 0.0014163, 0.0030967, 0.0012503, 0.0028775, -0.0009891, 0.0014507
4: 1.0022448, 1.0087645, 1.0016011, 1.0079139, -0.0038375, 0.0056280
5: 0.0026781, 0.0039464, 0.0025529, 0.0037810, -0.0007465, 0.0010949
6: -0.0108787, -0.0092281, -0.0106633, -0.0090652, -0.0014248, 0.0009715
7: -0.0101910, -0.0099805, -0.0101636, -0.0099597, -0.0001817, 0.0001239
8: -0.0044782, -0.0033378, -0.0045908, -0.0034866, -0.0006712, 0.0009844
9: -0.0014610, 0.0042481, -0.0007161, 0.0048118, -0.0049283, 0.0033604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024292, upper bound: 0.0024647
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025112, upper bound: 0.0024832
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040545, -0.0037321, -0.0040070, -0.0036657, -0.0003192, 0.0001835
1: 0.0004128, 0.0021978, 0.0000450, 0.0019349, -0.0010159, 0.0017672
2: 0.0100560, 0.0140438, 0.0106434, 0.0148656, -0.0039481, 0.0022696
3: 0.0014163, 0.0030967, 0.0010700, 0.0028492, -0.0009564, 0.0016638
4: 1.0022448, 1.0087645, 1.0009013, 1.0078042, -0.0037105, 0.0064547
5: 0.0026781, 0.0039464, 0.0024167, 0.0037596, -0.0007218, 0.0012557
6: -0.0108787, -0.0092281, -0.0106355, -0.0088880, -0.0016341, 0.0009394
7: -0.0101910, -0.0099805, -0.0101600, -0.0099371, -0.0002084, 0.0001198
8: -0.0044782, -0.0033378, -0.0047132, -0.0035058, -0.0006490, 0.0011290
9: -0.0014610, 0.0042481, -0.0006200, 0.0054246, -0.0056523, 0.0032492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0023112
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0024028
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040468, -0.0037003, -0.0040145, -0.0037066, -0.0002568, 0.0002241
1: 0.0002367, 0.0021552, 0.0002712, 0.0019762, -0.0012406, 0.0014218
2: 0.0101511, 0.0144372, 0.0105511, 0.0143603, -0.0031765, 0.0027716
3: 0.0012505, 0.0030566, 0.0012829, 0.0028881, -0.0011680, 0.0013386
4: 1.0016017, 1.0086088, 1.0017275, 1.0079550, -0.0045312, 0.0051932
5: 0.0025530, 0.0039162, 0.0025774, 0.0037890, -0.0008815, 0.0010103
6: -0.0108393, -0.0090653, -0.0106737, -0.0090971, -0.0013147, 0.0011471
7: -0.0101860, -0.0099597, -0.0101649, -0.0099638, -0.0001677, 0.0001463
8: -0.0045907, -0.0033650, -0.0045687, -0.0034794, -0.0007926, 0.0009084
9: -0.0013248, 0.0048113, -0.0007522, 0.0047012, -0.0045476, 0.0039679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022323, upper bound: 0.0024694
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022323, upper bound: 0.0024694
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040545, -0.0037321, -0.0040546, -0.0037268, -0.0002149, 0.0001965
1: 0.0004128, 0.0021978, 0.0003831, 0.0021982, -0.0010878, 0.0011899
2: 0.0100560, 0.0140438, 0.0100552, 0.0141103, -0.0026583, 0.0024303
3: 0.0014163, 0.0030967, 0.0013882, 0.0030971, -0.0010242, 0.0011202
4: 1.0022448, 1.0087645, 1.0021362, 1.0087657, -0.0039733, 0.0043461
5: 0.0026781, 0.0039464, 0.0026570, 0.0039467, -0.0007730, 0.0008455
6: -0.0108787, -0.0092281, -0.0108790, -0.0092006, -0.0011003, 0.0010059
7: -0.0101910, -0.0099805, -0.0101911, -0.0099770, -0.0001404, 0.0001283
8: -0.0044782, -0.0033378, -0.0044972, -0.0033376, -0.0006950, 0.0007602
9: -0.0014610, 0.0042481, -0.0014621, 0.0043433, -0.0038057, 0.0034793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024292, upper bound: 0.0024341
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025112, upper bound: 0.0024601
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040545, -0.0037321, -0.0040469, -0.0036951, -0.0002554, 0.0001900
1: 0.0004128, 0.0021978, 0.0002078, 0.0021555, -0.0010519, 0.0014144
2: 0.0100560, 0.0140438, 0.0101505, 0.0145019, -0.0031599, 0.0023500
3: 0.0014163, 0.0030967, 0.0012232, 0.0030569, -0.0009903, 0.0013316
4: 1.0022448, 1.0087645, 1.0014960, 1.0086099, -0.0038419, 0.0051660
5: 0.0026781, 0.0039464, 0.0025324, 0.0039164, -0.0007474, 0.0010050
6: -0.0108787, -0.0092281, -0.0108395, -0.0090385, -0.0013079, 0.0009726
7: -0.0101910, -0.0099805, -0.0101860, -0.0099563, -0.0001668, 0.0001241
8: -0.0044782, -0.0033378, -0.0046092, -0.0033649, -0.0006720, 0.0009036
9: -0.0014610, 0.0042481, -0.0013256, 0.0049039, -0.0045238, 0.0033643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0022627
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0023778
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040468, -0.0037003, -0.0040556, -0.0037325, -0.0001947, 0.0002385
1: 0.0002367, 0.0021552, 0.0004145, 0.0022037, -0.0013206, 0.0010779
2: 0.0101511, 0.0144372, 0.0100429, 0.0140400, -0.0024082, 0.0029504
3: 0.0012505, 0.0030566, 0.0014178, 0.0031023, -0.0012433, 0.0010148
4: 1.0016017, 1.0086088, 1.0022510, 1.0087858, -0.0048236, 0.0039371
5: 0.0025530, 0.0039162, 0.0026793, 0.0039506, -0.0009384, 0.0007659
6: -0.0108393, -0.0090653, -0.0108841, -0.0092297, -0.0009967, 0.0012212
7: -0.0101860, -0.0099597, -0.0101917, -0.0099807, -0.0001271, 0.0001558
8: -0.0045907, -0.0033650, -0.0044771, -0.0033341, -0.0008437, 0.0006887
9: -0.0013248, 0.0048113, -0.0014797, 0.0042427, -0.0034476, 0.0042239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022303, upper bound: 0.0024355
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022303, upper bound: 0.0024355
time: 0.81 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 7.60 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024239, upper bound: 0.0024925
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025404, upper bound: 0.0025322
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025036, upper bound: 0.0025505
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025563, upper bound: 0.0025563
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025164, upper bound: 0.0024431
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025164, upper bound: 0.0024432
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025195, upper bound: 0.0024477
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025195, upper bound: 0.0024807
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025152
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025152
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025159
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025159
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025145
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025145
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025611
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024431, upper bound: 0.0025611
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024469, upper bound: 0.0026135
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025075, upper bound: 0.0026479
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0023797, upper bound: 0.0023449
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0023797, upper bound: 0.0024332
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0022774, upper bound: 0.0024581
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0022774, upper bound: 0.0024580
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025713, upper bound: 0.0024656
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0026479, upper bound: 0.0025075
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0023820, upper bound: 0.0022403
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024766, upper bound: 0.0023005
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025469, upper bound: 0.0024925
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0026479, upper bound: 0.0025075
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024300, upper bound: 0.0023021
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024300, upper bound: 0.0023688
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024292, upper bound: 0.0024647
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025112, upper bound: 0.0024832
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0023112
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0024028
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0022323, upper bound: 0.0024694
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0022323, upper bound: 0.0024694
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0024292, upper bound: 0.0024341
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0025112, upper bound: 0.0024601
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0022627
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0023548, upper bound: 0.0023778
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0022303, upper bound: 0.0024355
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.60
Output dim: 4, lower bound: -0.0022303, upper bound: 0.0024355

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040211, -0.0037142, -0.0039958, -0.0036956, -0.0001980, 0.0001679
1: 0.0003136, 0.0020128, 0.0002104, 0.0018729, -0.0009299, 0.0010961
2: 0.0104693, 0.0142655, 0.0107818, 0.0144961, -0.0024488, 0.0020775
3: 0.0013228, 0.0029226, 0.0012257, 0.0027909, -0.0008755, 0.0010319
4: 1.0018824, 1.0080888, 1.0015054, 1.0075778, -0.0033965, 0.0040035
5: 0.0026076, 0.0038150, 0.0025343, 0.0037156, -0.0006607, 0.0007788
6: -0.0107076, -0.0091364, -0.0105783, -0.0090409, -0.0010135, 0.0008599
7: -0.0101692, -0.0099688, -0.0101527, -0.0099566, -0.0001293, 0.0001097
8: -0.0045416, -0.0034560, -0.0046075, -0.0035454, -0.0005941, 0.0007003
9: -0.0008693, 0.0045655, -0.0004219, 0.0048956, -0.0035057, 0.0029742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024239, upper bound: 0.0024064
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024239, upper bound: 0.0024925
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040265, -0.0037136, -0.0040090, -0.0037063, -0.0002043, 0.0001688
1: 0.0003100, 0.0020427, 0.0002696, 0.0019460, -0.0009344, 0.0011314
2: 0.0104026, 0.0142737, 0.0106186, 0.0143639, -0.0025276, 0.0020875
3: 0.0013194, 0.0029507, 0.0012814, 0.0028597, -0.0008797, 0.0010652
4: 1.0018691, 1.0081978, 1.0017216, 1.0078447, -0.0034128, 0.0041324
5: 0.0026050, 0.0038362, 0.0025763, 0.0037675, -0.0006639, 0.0008039
6: -0.0107352, -0.0091330, -0.0106458, -0.0090956, -0.0010462, 0.0008640
7: -0.0101727, -0.0099684, -0.0101613, -0.0099636, -0.0001335, 0.0001102
8: -0.0045439, -0.0034370, -0.0045697, -0.0034987, -0.0005970, 0.0007228
9: -0.0009647, 0.0045772, -0.0006555, 0.0047063, -0.0036186, 0.0029885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024869, upper bound: 0.0024064
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024869, upper bound: 0.0025322
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040064, -0.0037069, -0.0039959, -0.0036904, -0.0002017, 0.0001710
1: 0.0002730, 0.0019312, 0.0001815, 0.0018734, -0.0009471, 0.0011167
2: 0.0106516, 0.0143563, 0.0107808, 0.0145607, -0.0024949, 0.0021159
3: 0.0012846, 0.0028457, 0.0011985, 0.0027913, -0.0008916, 0.0010514
4: 1.0017340, 1.0077907, 1.0013999, 1.0075794, -0.0034592, 0.0040789
5: 0.0025787, 0.0037570, 0.0025137, 0.0037159, -0.0006730, 0.0007935
6: -0.0106321, -0.0090988, -0.0105787, -0.0090142, -0.0010326, 0.0008758
7: -0.0101596, -0.0099640, -0.0101528, -0.0099532, -0.0001317, 0.0001117
8: -0.0045676, -0.0035082, -0.0046260, -0.0035451, -0.0006051, 0.0007135
9: -0.0006082, 0.0046955, -0.0004232, 0.0049880, -0.0035718, 0.0030292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021930, upper bound: 0.0021827
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020871, upper bound: 0.0021509
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040123, -0.0037063, -0.0040091, -0.0037007, -0.0002088, 0.0001732
1: 0.0002695, 0.0019641, 0.0002390, 0.0019464, -0.0009591, 0.0011559
2: 0.0105781, 0.0143640, 0.0106177, 0.0144323, -0.0025823, 0.0021426
3: 0.0012813, 0.0028767, 0.0012525, 0.0028600, -0.0009029, 0.0010882
4: 1.0017214, 1.0079108, 1.0016097, 1.0078462, -0.0035030, 0.0042218
5: 0.0025763, 0.0037804, 0.0025545, 0.0037678, -0.0006815, 0.0008213
6: -0.0106625, -0.0090956, -0.0106462, -0.0090673, -0.0010688, 0.0008868
7: -0.0101635, -0.0099636, -0.0101614, -0.0099600, -0.0001363, 0.0001131
8: -0.0045698, -0.0034871, -0.0045893, -0.0034984, -0.0006127, 0.0007385
9: -0.0007134, 0.0047064, -0.0006568, 0.0048043, -0.0036969, 0.0030675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025505, upper bound: 0.0025036
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025505, upper bound: 0.0025563
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040266, -0.0037136, -0.0040202, -0.0036791, -0.0002370, 0.0001864
1: 0.0003099, 0.0020432, 0.0001189, 0.0020078, -0.0010323, 0.0013120
2: 0.0104015, 0.0142738, 0.0104806, 0.0147006, -0.0029311, 0.0023063
3: 0.0013193, 0.0029511, 0.0011395, 0.0029178, -0.0009719, 0.0012352
4: 1.0018688, 1.0081997, 1.0011711, 1.0080702, -0.0037705, 0.0047920
5: 0.0026050, 0.0038366, 0.0024692, 0.0038114, -0.0007335, 0.0009322
6: -0.0107357, -0.0091329, -0.0107029, -0.0089563, -0.0012132, 0.0009546
7: -0.0101728, -0.0099683, -0.0101686, -0.0099458, -0.0001548, 0.0001218
8: -0.0045440, -0.0034366, -0.0046660, -0.0034593, -0.0006595, 0.0008382
9: -0.0009664, 0.0045774, -0.0008531, 0.0051884, -0.0041963, 0.0033017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019857, upper bound: 0.0016709
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0016292, upper bound: 0.0015215
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040266, -0.0037136, -0.0040070, -0.0036710, -0.0002521, 0.0001826
1: 0.0003099, 0.0020432, 0.0000745, 0.0019346, -0.0010108, 0.0013959
2: 0.0104015, 0.0142738, 0.0106440, 0.0147997, -0.0031186, 0.0022582
3: 0.0013193, 0.0029511, 0.0010977, 0.0028490, -0.0009516, 0.0013142
4: 1.0018688, 1.0081997, 1.0010091, 1.0078032, -0.0036919, 0.0050985
5: 0.0026050, 0.0038366, 0.0024377, 0.0037594, -0.0007182, 0.0009919
6: -0.0107357, -0.0091329, -0.0106353, -0.0089153, -0.0012908, 0.0009347
7: -0.0101728, -0.0099683, -0.0101600, -0.0099406, -0.0001646, 0.0001192
8: -0.0045440, -0.0034366, -0.0046944, -0.0035060, -0.0006458, 0.0008918
9: -0.0009664, 0.0045774, -0.0006192, 0.0053303, -0.0044646, 0.0032329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019857, upper bound: 0.0019108
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0016292, upper bound: 0.0017547
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040124, -0.0037063, -0.0040202, -0.0036791, -0.0002319, 0.0002053
1: 0.0002695, 0.0019646, 0.0001189, 0.0020078, -0.0011366, 0.0012843
2: 0.0105771, 0.0143641, 0.0104806, 0.0147006, -0.0028692, 0.0025392
3: 0.0012813, 0.0028772, 0.0011395, 0.0029178, -0.0010700, 0.0012091
4: 1.0017211, 1.0079125, 1.0011711, 1.0080702, -0.0041513, 0.0046909
5: 0.0025762, 0.0037807, 0.0024692, 0.0038114, -0.0008076, 0.0009126
6: -0.0106630, -0.0090956, -0.0107029, -0.0089563, -0.0011876, 0.0010510
7: -0.0101635, -0.0099636, -0.0101686, -0.0099458, -0.0001515, 0.0001341
8: -0.0045698, -0.0034868, -0.0046660, -0.0034593, -0.0007261, 0.0008205
9: -0.0007150, 0.0047066, -0.0008531, 0.0051884, -0.0041077, 0.0036352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020275, upper bound: 0.0017145
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018101, upper bound: 0.0015976
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040124, -0.0037063, -0.0040070, -0.0036710, -0.0002327, 0.0001838
1: 0.0002695, 0.0019646, 0.0000745, 0.0019346, -0.0010177, 0.0012887
2: 0.0105771, 0.0143641, 0.0106440, 0.0147997, -0.0028790, 0.0022736
3: 0.0012813, 0.0028772, 0.0010977, 0.0028490, -0.0009581, 0.0012132
4: 1.0017211, 1.0079125, 1.0010091, 1.0078032, -0.0037171, 0.0047069
5: 0.0025762, 0.0037807, 0.0024377, 0.0037594, -0.0007231, 0.0009157
6: -0.0106630, -0.0090956, -0.0106353, -0.0089153, -0.0011916, 0.0009410
7: -0.0101635, -0.0099636, -0.0101600, -0.0099406, -0.0001520, 0.0001200
8: -0.0045698, -0.0034868, -0.0046944, -0.0035060, -0.0006502, 0.0008233
9: -0.0007150, 0.0047066, -0.0006192, 0.0053303, -0.0041217, 0.0032550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020275, upper bound: 0.0021085
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018101, upper bound: 0.0020682
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040202, -0.0036791, -0.0040266, -0.0037136, -0.0001864, 0.0002370
1: 0.0001189, 0.0020078, 0.0003099, 0.0020432, -0.0013120, 0.0010323
2: 0.0104806, 0.0147006, 0.0104015, 0.0142738, -0.0023063, 0.0029311
3: 0.0011395, 0.0029178, 0.0013193, 0.0029511, -0.0012352, 0.0009719
4: 1.0011711, 1.0080702, 1.0018688, 1.0081997, -0.0047920, 0.0037705
5: 0.0024692, 0.0038114, 0.0026050, 0.0038366, -0.0009322, 0.0007335
6: -0.0107029, -0.0089563, -0.0107357, -0.0091329, -0.0009546, 0.0012132
7: -0.0101686, -0.0099458, -0.0101728, -0.0099683, -0.0001218, 0.0001548
8: -0.0046660, -0.0034593, -0.0045440, -0.0034366, -0.0008382, 0.0006595
9: -0.0008531, 0.0051884, -0.0009664, 0.0045774, -0.0033017, 0.0041963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018682, upper bound: 0.0016457
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0014101, upper bound: 0.0014101
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040202, -0.0036791, -0.0040202, -0.0036791, -0.0002109, 0.0002109
1: 0.0001189, 0.0020078, 0.0001189, 0.0020078, -0.0011675, 0.0011675
2: 0.0104806, 0.0147006, 0.0104806, 0.0147006, -0.0026083, 0.0026083
3: 0.0011395, 0.0029178, 0.0011395, 0.0029178, -0.0010991, 0.0010991
4: 1.0011711, 1.0080702, 1.0011711, 1.0080702, -0.0042642, 0.0042642
5: 0.0024692, 0.0038114, 0.0024692, 0.0038114, -0.0008296, 0.0008296
6: -0.0107029, -0.0089563, -0.0107029, -0.0089563, -0.0010796, 0.0010796
7: -0.0101686, -0.0099458, -0.0101686, -0.0099458, -0.0001377, 0.0001377
8: -0.0046660, -0.0034593, -0.0046660, -0.0034593, -0.0007459, 0.0007459
9: -0.0008531, 0.0051884, -0.0008531, 0.0051884, -0.0037341, 0.0037341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018682, upper bound: 0.0016457
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0014101, upper bound: 0.0014101
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040266, -0.0037136, -0.0001826, 0.0002521
1: 0.0000745, 0.0019346, 0.0003099, 0.0020432, -0.0013959, 0.0010108
2: 0.0106440, 0.0147997, 0.0104015, 0.0142738, -0.0022582, 0.0031186
3: 0.0010977, 0.0028490, 0.0013193, 0.0029511, -0.0013142, 0.0009516
4: 1.0010091, 1.0078032, 1.0018688, 1.0081997, -0.0050985, 0.0036919
5: 0.0024377, 0.0037594, 0.0026050, 0.0038366, -0.0009919, 0.0007182
6: -0.0106353, -0.0089153, -0.0107357, -0.0091329, -0.0009347, 0.0012908
7: -0.0101600, -0.0099406, -0.0101728, -0.0099683, -0.0001192, 0.0001646
8: -0.0046944, -0.0035060, -0.0045440, -0.0034366, -0.0008918, 0.0006458
9: -0.0006192, 0.0053303, -0.0009664, 0.0045774, -0.0032329, 0.0044646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018969, upper bound: 0.0016881
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0016170, upper bound: 0.0015372
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040202, -0.0036791, -0.0002068, 0.0002274
1: 0.0000745, 0.0019346, 0.0001189, 0.0020078, -0.0012590, 0.0011449
2: 0.0106440, 0.0147997, 0.0104806, 0.0147006, -0.0025579, 0.0028126
3: 0.0010977, 0.0028490, 0.0011395, 0.0029178, -0.0011853, 0.0010779
4: 1.0010091, 1.0078032, 1.0011711, 1.0080702, -0.0045983, 0.0041818
5: 0.0024377, 0.0037594, 0.0024692, 0.0038114, -0.0008946, 0.0008135
6: -0.0106353, -0.0089153, -0.0107029, -0.0089563, -0.0010587, 0.0011641
7: -0.0101600, -0.0099406, -0.0101686, -0.0099458, -0.0001350, 0.0001485
8: -0.0046944, -0.0035060, -0.0046660, -0.0034593, -0.0008043, 0.0007315
9: -0.0006192, 0.0053303, -0.0008531, 0.0051884, -0.0036619, 0.0040266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018969, upper bound: 0.0016881
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0016170, upper bound: 0.0015372
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040202, -0.0036791, -0.0040124, -0.0037063, -0.0002053, 0.0002319
1: 0.0001189, 0.0020078, 0.0002695, 0.0019646, -0.0012843, 0.0011366
2: 0.0104806, 0.0147006, 0.0105771, 0.0143641, -0.0025392, 0.0028692
3: 0.0011395, 0.0029178, 0.0012813, 0.0028772, -0.0012091, 0.0010700
4: 1.0011711, 1.0080702, 1.0017211, 1.0079125, -0.0046909, 0.0041513
5: 0.0024692, 0.0038114, 0.0025762, 0.0037807, -0.0009126, 0.0008076
6: -0.0107029, -0.0089563, -0.0106630, -0.0090956, -0.0010510, 0.0011876
7: -0.0101686, -0.0099458, -0.0101635, -0.0099636, -0.0001341, 0.0001515
8: -0.0046660, -0.0034593, -0.0045698, -0.0034868, -0.0008205, 0.0007261
9: -0.0008531, 0.0051884, -0.0007150, 0.0047066, -0.0036352, 0.0041077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020155, upper bound: 0.0018977
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015372, upper bound: 0.0016170
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040202, -0.0036791, -0.0040070, -0.0036710, -0.0002274, 0.0002068
1: 0.0001189, 0.0020078, 0.0000745, 0.0019346, -0.0011449, 0.0012590
2: 0.0104806, 0.0147006, 0.0106440, 0.0147997, -0.0028126, 0.0025579
3: 0.0011395, 0.0029178, 0.0010977, 0.0028490, -0.0010779, 0.0011853
4: 1.0011711, 1.0080702, 1.0010091, 1.0078032, -0.0041818, 0.0045983
5: 0.0024692, 0.0038114, 0.0024377, 0.0037594, -0.0008135, 0.0008946
6: -0.0107029, -0.0089563, -0.0106353, -0.0089153, -0.0011641, 0.0010587
7: -0.0101686, -0.0099458, -0.0101600, -0.0099406, -0.0001485, 0.0001350
8: -0.0046660, -0.0034593, -0.0046944, -0.0035060, -0.0007315, 0.0008043
9: -0.0008531, 0.0051884, -0.0006192, 0.0053303, -0.0040266, 0.0036619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020155, upper bound: 0.0018977
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015372, upper bound: 0.0016170
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040124, -0.0037063, -0.0001838, 0.0002327
1: 0.0000745, 0.0019346, 0.0002695, 0.0019646, -0.0012887, 0.0010177
2: 0.0106440, 0.0147997, 0.0105771, 0.0143641, -0.0022736, 0.0028790
3: 0.0010977, 0.0028490, 0.0012813, 0.0028772, -0.0012132, 0.0009581
4: 1.0010091, 1.0078032, 1.0017211, 1.0079125, -0.0047069, 0.0037171
5: 0.0024377, 0.0037594, 0.0025762, 0.0037807, -0.0009157, 0.0007231
6: -0.0106353, -0.0089153, -0.0106630, -0.0090956, -0.0009410, 0.0011916
7: -0.0101600, -0.0099406, -0.0101635, -0.0099636, -0.0001200, 0.0001520
8: -0.0046944, -0.0035060, -0.0045698, -0.0034868, -0.0008233, 0.0006502
9: -0.0006192, 0.0053303, -0.0007150, 0.0047066, -0.0032550, 0.0041217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021769, upper bound: 0.0021091
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020575, upper bound: 0.0020550
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040070, -0.0036710, -0.0002072, 0.0002072
1: 0.0000745, 0.0019346, 0.0000745, 0.0019346, -0.0011471, 0.0011471
2: 0.0106440, 0.0147997, 0.0106440, 0.0147997, -0.0025627, 0.0025627
3: 0.0010977, 0.0028490, 0.0010977, 0.0028490, -0.0010799, 0.0010799
4: 1.0010091, 1.0078032, 1.0010091, 1.0078032, -0.0041898, 0.0041898
5: 0.0024377, 0.0037594, 0.0024377, 0.0037594, -0.0008151, 0.0008151
6: -0.0106353, -0.0089153, -0.0106353, -0.0089153, -0.0010607, 0.0010607
7: -0.0101600, -0.0099406, -0.0101600, -0.0099406, -0.0001353, 0.0001353
8: -0.0046944, -0.0035060, -0.0046944, -0.0035060, -0.0007329, 0.0007329
9: -0.0006192, 0.0053303, -0.0006192, 0.0053303, -0.0036689, 0.0036689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0021769, upper bound: 0.0021091
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020575, upper bound: 0.0020550
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040064, -0.0037069, -0.0040398, -0.0036991, -0.0002061, 0.0002289
1: 0.0002730, 0.0019312, 0.0002301, 0.0021162, -0.0012676, 0.0011410
2: 0.0106516, 0.0143563, 0.0102383, 0.0144521, -0.0025491, 0.0028321
3: 0.0012846, 0.0028457, 0.0012442, 0.0030199, -0.0011934, 0.0010742
4: 1.0017340, 1.0077907, 1.0015774, 1.0084664, -0.0046301, 0.0041675
5: 0.0025787, 0.0037570, 0.0025483, 0.0038885, -0.0009007, 0.0008107
6: -0.0106321, -0.0090988, -0.0108032, -0.0090591, -0.0010551, 0.0011722
7: -0.0101596, -0.0099640, -0.0101814, -0.0099589, -0.0001346, 0.0001495
8: -0.0045676, -0.0035082, -0.0045950, -0.0033900, -0.0008099, 0.0007290
9: -0.0006082, 0.0046955, -0.0012000, 0.0048326, -0.0036493, 0.0040544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018479, upper bound: 0.0017196
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015384, upper bound: 0.0015956
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040123, -0.0037063, -0.0040544, -0.0037074, -0.0002105, 0.0002308
1: 0.0002695, 0.0019641, 0.0002757, 0.0021970, -0.0012779, 0.0011653
2: 0.0105781, 0.0143640, 0.0100578, 0.0143502, -0.0026035, 0.0028551
3: 0.0012813, 0.0028767, 0.0012871, 0.0030960, -0.0012031, 0.0010971
4: 1.0017214, 1.0079108, 1.0017439, 1.0087614, -0.0046677, 0.0042564
5: 0.0025763, 0.0037804, 0.0025807, 0.0039459, -0.0009080, 0.0008280
6: -0.0106625, -0.0090956, -0.0108779, -0.0091013, -0.0010776, 0.0011817
7: -0.0101635, -0.0099636, -0.0101909, -0.0099643, -0.0001375, 0.0001507
8: -0.0045698, -0.0034871, -0.0045658, -0.0033383, -0.0008165, 0.0007445
9: -0.0007134, 0.0047064, -0.0014583, 0.0046867, -0.0037272, 0.0040874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024925, upper bound: 0.0025469
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024925, upper bound: 0.0026479
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040124, -0.0037063, -0.0040506, -0.0036797, -0.0002340, 0.0002382
1: 0.0002695, 0.0019646, 0.0001225, 0.0021762, -0.0013188, 0.0012958
2: 0.0105771, 0.0143641, 0.0101043, 0.0146926, -0.0028950, 0.0029463
3: 0.0012813, 0.0028772, 0.0011429, 0.0030764, -0.0012416, 0.0012200
4: 1.0017211, 1.0079125, 1.0011842, 1.0086854, -0.0048169, 0.0047330
5: 0.0025762, 0.0037807, 0.0024718, 0.0039311, -0.0009371, 0.0009208
6: -0.0106630, -0.0090956, -0.0108587, -0.0089596, -0.0011982, 0.0012195
7: -0.0101635, -0.0099636, -0.0101885, -0.0099462, -0.0001528, 0.0001556
8: -0.0045698, -0.0034868, -0.0046637, -0.0033516, -0.0008426, 0.0008279
9: -0.0007150, 0.0047066, -0.0013917, 0.0051769, -0.0041446, 0.0042180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015208, upper bound: 0.0018188
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023337, upper bound: 0.0023888
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040575, -0.0037128, -0.0001885, 0.0002893
1: 0.0000745, 0.0019346, 0.0003056, 0.0022145, -0.0016016, 0.0010437
2: 0.0106440, 0.0147997, 0.0100186, 0.0142833, -0.0023318, 0.0035781
3: 0.0010977, 0.0028490, 0.0013153, 0.0031125, -0.0015078, 0.0009826
4: 1.0010091, 1.0078032, 1.0018532, 1.0088255, -0.0058498, 0.0038122
5: 0.0024377, 0.0037594, 0.0026019, 0.0039583, -0.0011380, 0.0007416
6: -0.0106353, -0.0089153, -0.0108941, -0.0091290, -0.0009651, 0.0014810
7: -0.0101600, -0.0099406, -0.0101930, -0.0099678, -0.0001231, 0.0001889
8: -0.0046944, -0.0035060, -0.0045467, -0.0033271, -0.0010232, 0.0006668
9: -0.0006192, 0.0053303, -0.0015144, 0.0045910, -0.0033382, 0.0051226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015351, upper bound: 0.0016133
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023271, upper bound: 0.0024141
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040070, -0.0036710, -0.0040506, -0.0036797, -0.0002098, 0.0002639
1: 0.0000745, 0.0019346, 0.0001225, 0.0021762, -0.0014610, 0.0011616
2: 0.0106440, 0.0147997, 0.0101043, 0.0146926, -0.0025951, 0.0032641
3: 0.0010977, 0.0028490, 0.0011429, 0.0030764, -0.0013755, 0.0010936
4: 1.0010091, 1.0078032, 1.0011842, 1.0086854, -0.0053364, 0.0042426
5: 0.0024377, 0.0037594, 0.0024718, 0.0039311, -0.0010381, 0.0008254
6: -0.0106353, -0.0089153, -0.0108587, -0.0089596, -0.0010741, 0.0013510
7: -0.0101600, -0.0099406, -0.0101885, -0.0099462, -0.0001370, 0.0001723
8: -0.0046944, -0.0035060, -0.0046637, -0.0033516, -0.0009334, 0.0007421
9: -0.0006192, 0.0053303, -0.0013917, 0.0051769, -0.0037151, 0.0046729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015351, upper bound: 0.0016133
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023271, upper bound: 0.0024141
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040543, -0.0037133, -0.0040265, -0.0037136, -0.0002247, 0.0002077
1: 0.0003083, 0.0021966, 0.0003100, 0.0020427, -0.0011499, 0.0012443
2: 0.0100588, 0.0142774, 0.0104026, 0.0142737, -0.0027800, 0.0025690
3: 0.0013178, 0.0030956, 0.0013194, 0.0029507, -0.0010826, 0.0011715
4: 1.0018630, 1.0087599, 1.0018691, 1.0081978, -0.0042001, 0.0045449
5: 0.0026038, 0.0039455, 0.0026050, 0.0038362, -0.0008171, 0.0008842
6: -0.0108775, -0.0091315, -0.0107352, -0.0091330, -0.0011506, 0.0010633
7: -0.0101909, -0.0099682, -0.0101727, -0.0099684, -0.0001468, 0.0001356
8: -0.0045450, -0.0033386, -0.0045439, -0.0034370, -0.0007347, 0.0007950
9: -0.0014570, 0.0045825, -0.0009647, 0.0045772, -0.0039799, 0.0036779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025581, upper bound: 0.0024605
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025581, upper bound: 0.0024621
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040544, -0.0037074, -0.0040123, -0.0037063, -0.0002308, 0.0002105
1: 0.0002757, 0.0021970, 0.0002695, 0.0019641, -0.0011653, 0.0012779
2: 0.0100578, 0.0143502, 0.0105781, 0.0143640, -0.0028551, 0.0026035
3: 0.0012871, 0.0030960, 0.0012813, 0.0028767, -0.0010971, 0.0012031
4: 1.0017439, 1.0087614, 1.0017214, 1.0079108, -0.0042564, 0.0046677
5: 0.0025807, 0.0039459, 0.0025763, 0.0037804, -0.0008280, 0.0009080
6: -0.0108779, -0.0091013, -0.0106625, -0.0090956, -0.0011817, 0.0010776
7: -0.0101909, -0.0099643, -0.0101635, -0.0099636, -0.0001507, 0.0001375
8: -0.0045658, -0.0033383, -0.0045698, -0.0034871, -0.0007445, 0.0008165
9: -0.0014583, 0.0046867, -0.0007134, 0.0047064, -0.0040874, 0.0037272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025648, upper bound: 0.0024654
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0025648, upper bound: 0.0025075
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040544, -0.0037074, -0.0040069, -0.0036711, -0.0002694, 0.0002051
1: 0.0002757, 0.0021970, 0.0000746, 0.0019343, -0.0011356, 0.0014917
2: 0.0100578, 0.0143502, 0.0106448, 0.0147996, -0.0033326, 0.0025370
3: 0.0012871, 0.0030960, 0.0010978, 0.0028486, -0.0010691, 0.0014044
4: 1.0017439, 1.0087614, 1.0010093, 1.0078018, -0.0041476, 0.0054484
5: 0.0025807, 0.0039459, 0.0024377, 0.0037592, -0.0008069, 0.0010599
6: -0.0108779, -0.0091013, -0.0106349, -0.0089153, -0.0013793, 0.0010500
7: -0.0101909, -0.0099643, -0.0101599, -0.0099406, -0.0001759, 0.0001339
8: -0.0045658, -0.0033383, -0.0046943, -0.0035062, -0.0007255, 0.0009530
9: -0.0014583, 0.0046867, -0.0006180, 0.0053301, -0.0047710, 0.0036320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023739, upper bound: 0.0022322
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0023739, upper bound: 0.0023005
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040510, -0.0037135, -0.0040398, -0.0036991, -0.0002121, 0.0001821
1: 0.0003095, 0.0021786, 0.0002301, 0.0021162, -0.0010083, 0.0011742
2: 0.0100989, 0.0142746, 0.0102383, 0.0144521, -0.0026233, 0.0022527
3: 0.0013190, 0.0030786, 0.0012442, 0.0030199, -0.0009493, 0.0011055
4: 1.0018675, 1.0086942, 1.0015774, 1.0084664, -0.0036830, 0.0042887
5: 0.0026047, 0.0039328, 0.0025483, 0.0038885, -0.0007165, 0.0008343
6: -0.0108609, -0.0091326, -0.0108032, -0.0090591, -0.0010858, 0.0009324
7: -0.0101888, -0.0099683, -0.0101814, -0.0099589, -0.0001385, 0.0001189
8: -0.0045442, -0.0033501, -0.0045950, -0.0033900, -0.0006442, 0.0007502
9: -0.0013995, 0.0045785, -0.0012000, 0.0048326, -0.0037555, 0.0032251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018138, upper bound: 0.0016339
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0013867, upper bound: 0.0014932
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040575, -0.0037128, -0.0040544, -0.0037074, -0.0002192, 0.0001849
1: 0.0003057, 0.0022141, 0.0002757, 0.0021970, -0.0010237, 0.0012139
2: 0.0100196, 0.0142832, 0.0100578, 0.0143502, -0.0027121, 0.0022871
3: 0.0013154, 0.0031121, 0.0012871, 0.0030960, -0.0009638, 0.0011429
4: 1.0018535, 1.0088239, 1.0017439, 1.0087614, -0.0037391, 0.0044339
5: 0.0026020, 0.0039580, 0.0025807, 0.0039459, -0.0007274, 0.0008626
6: -0.0108937, -0.0091291, -0.0108779, -0.0091013, -0.0011225, 0.0009466
7: -0.0101930, -0.0099679, -0.0101909, -0.0099643, -0.0001432, 0.0001207
8: -0.0045467, -0.0033274, -0.0045658, -0.0033383, -0.0006540, 0.0007756
9: -0.0015130, 0.0045908, -0.0014583, 0.0046867, -0.0038826, 0.0032742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026135, upper bound: 0.0024464
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026135, upper bound: 0.0025075
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040491, -0.0037328, -0.0039959, -0.0036904, -0.0002718, 0.0001706
1: 0.0004167, 0.0021679, 0.0001815, 0.0018734, -0.0009445, 0.0015047
2: 0.0101229, 0.0140351, 0.0107808, 0.0145607, -0.0033617, 0.0021101
3: 0.0014199, 0.0030686, 0.0011985, 0.0027913, -0.0008892, 0.0014166
4: 1.0022590, 1.0086552, 1.0013999, 1.0075794, -0.0034497, 0.0054960
5: 0.0026809, 0.0039252, 0.0025137, 0.0037159, -0.0006711, 0.0010692
6: -0.0108510, -0.0092317, -0.0105787, -0.0090142, -0.0013914, 0.0008733
7: -0.0101875, -0.0099809, -0.0101528, -0.0099532, -0.0001775, 0.0001114
8: -0.0044757, -0.0033569, -0.0046260, -0.0035451, -0.0006034, 0.0009613
9: -0.0013652, 0.0042357, -0.0004232, 0.0049880, -0.0048127, 0.0030208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0017526, upper bound: 0.0018133
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0014538, upper bound: 0.0016623
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040544, -0.0037322, -0.0040091, -0.0037007, -0.0002777, 0.0001795
1: 0.0004129, 0.0021973, 0.0002390, 0.0019464, -0.0009939, 0.0015377
2: 0.0100572, 0.0140437, 0.0106177, 0.0144323, -0.0034353, 0.0022206
3: 0.0014163, 0.0030962, 0.0012525, 0.0028600, -0.0009357, 0.0014477
4: 1.0022451, 1.0087624, 1.0016097, 1.0078462, -0.0036303, 0.0056164
5: 0.0026782, 0.0039460, 0.0025545, 0.0037678, -0.0007062, 0.0010926
6: -0.0108781, -0.0092282, -0.0106462, -0.0090673, -0.0014219, 0.0009191
7: -0.0101910, -0.0099805, -0.0101614, -0.0099600, -0.0001814, 0.0001172
8: -0.0044782, -0.0033382, -0.0045893, -0.0034984, -0.0006350, 0.0009824
9: -0.0014592, 0.0042479, -0.0006568, 0.0048043, -0.0049181, 0.0031790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0024036
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0024832
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040468, -0.0037003, -0.0040124, -0.0037063, -0.0002471, 0.0002244
1: 0.0002367, 0.0021552, 0.0002695, 0.0019646, -0.0012427, 0.0013683
2: 0.0101511, 0.0144372, 0.0105771, 0.0143641, -0.0030570, 0.0027764
3: 0.0012505, 0.0030566, 0.0012813, 0.0028772, -0.0011700, 0.0012882
4: 1.0016017, 1.0086088, 1.0017211, 1.0079125, -0.0045391, 0.0049979
5: 0.0025530, 0.0039162, 0.0025762, 0.0037807, -0.0008830, 0.0009723
6: -0.0108393, -0.0090653, -0.0106630, -0.0090956, -0.0012653, 0.0011492
7: -0.0101860, -0.0099597, -0.0101635, -0.0099636, -0.0001614, 0.0001466
8: -0.0045907, -0.0033650, -0.0045698, -0.0034868, -0.0007940, 0.0008742
9: -0.0013248, 0.0048113, -0.0007150, 0.0047066, -0.0043765, 0.0039748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018930, upper bound: 0.0018913
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0016351, upper bound: 0.0017355
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040468, -0.0037003, -0.0040070, -0.0036710, -0.0002753, 0.0002022
1: 0.0002367, 0.0021552, 0.0000745, 0.0019346, -0.0011196, 0.0015243
2: 0.0101511, 0.0144372, 0.0106440, 0.0147997, -0.0034055, 0.0025012
3: 0.0012505, 0.0030566, 0.0010977, 0.0028490, -0.0010540, 0.0014351
4: 1.0016017, 1.0086088, 1.0010091, 1.0078032, -0.0040892, 0.0055675
5: 0.0025530, 0.0039162, 0.0024377, 0.0037594, -0.0007955, 0.0010831
6: -0.0108393, -0.0090653, -0.0106353, -0.0089153, -0.0014095, 0.0010352
7: -0.0101860, -0.0099597, -0.0101600, -0.0099406, -0.0001798, 0.0001321
8: -0.0045907, -0.0033650, -0.0046944, -0.0035060, -0.0007153, 0.0009739
9: -0.0013248, 0.0048113, -0.0006192, 0.0053303, -0.0048753, 0.0035808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018930, upper bound: 0.0018913
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0016351, upper bound: 0.0017355
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040491, -0.0037328, -0.0040407, -0.0037134, -0.0002069, 0.0001777
1: 0.0004167, 0.0021679, 0.0003089, 0.0021216, -0.0009838, 0.0011455
2: 0.0101229, 0.0140351, 0.0102264, 0.0142759, -0.0025593, 0.0021979
3: 0.0014199, 0.0030686, 0.0013184, 0.0030249, -0.0009262, 0.0010785
4: 1.0022590, 1.0086552, 1.0018654, 1.0084859, -0.0035933, 0.0041841
5: 0.0026809, 0.0039252, 0.0026043, 0.0038922, -0.0006990, 0.0008140
6: -0.0108510, -0.0092317, -0.0108081, -0.0091320, -0.0010593, 0.0009097
7: -0.0101875, -0.0099809, -0.0101820, -0.0099682, -0.0001351, 0.0001160
8: -0.0044757, -0.0033569, -0.0045446, -0.0033865, -0.0006285, 0.0007319
9: -0.0013652, 0.0042357, -0.0012170, 0.0045804, -0.0036639, 0.0031466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024275, upper bound: 0.0023834
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024275, upper bound: 0.0024341
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040544, -0.0037322, -0.0040507, -0.0037272, -0.0002143, 0.0001777
1: 0.0004129, 0.0021973, 0.0003856, 0.0021765, -0.0009838, 0.0011867
2: 0.0100572, 0.0140437, 0.0101037, 0.0141046, -0.0026512, 0.0021979
3: 0.0014163, 0.0030962, 0.0013906, 0.0030766, -0.0009262, 0.0011172
4: 1.0022451, 1.0087624, 1.0021455, 1.0086865, -0.0035933, 0.0043343
5: 0.0026782, 0.0039460, 0.0026588, 0.0039313, -0.0006990, 0.0008432
6: -0.0108781, -0.0092282, -0.0108589, -0.0092030, -0.0010973, 0.0009097
7: -0.0101910, -0.0099805, -0.0101885, -0.0099773, -0.0001400, 0.0001160
8: -0.0044782, -0.0033382, -0.0044956, -0.0033515, -0.0006285, 0.0007581
9: -0.0014592, 0.0042479, -0.0013927, 0.0043351, -0.0037955, 0.0031465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0023834
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0024601
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040468, -0.0037003, -0.0040545, -0.0037321, -0.0001899, 0.0002388
1: 0.0002367, 0.0021552, 0.0004128, 0.0021978, -0.0013221, 0.0010516
2: 0.0101511, 0.0144372, 0.0100560, 0.0140438, -0.0023495, 0.0029537
3: 0.0012505, 0.0030566, 0.0014163, 0.0030967, -0.0012447, 0.0009901
4: 1.0016017, 1.0086088, 1.0022448, 1.0087645, -0.0048289, 0.0038412
5: 0.0025530, 0.0039162, 0.0026781, 0.0039464, -0.0009394, 0.0007473
6: -0.0108393, -0.0090653, -0.0108787, -0.0092281, -0.0009724, 0.0012225
7: -0.0101860, -0.0099597, -0.0101910, -0.0099805, -0.0001240, 0.0001559
8: -0.0045907, -0.0033650, -0.0044782, -0.0033378, -0.0008447, 0.0006719
9: -0.0013248, 0.0048113, -0.0014610, 0.0042481, -0.0033636, 0.0042286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018313, upper bound: 0.0017036
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015491, upper bound: 0.0015491
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040468, -0.0037003, -0.0040468, -0.0037003, -0.0002129, 0.0002129
1: 0.0002367, 0.0021552, 0.0002367, 0.0021552, -0.0011788, 0.0011788
2: 0.0101511, 0.0144372, 0.0101511, 0.0144372, -0.0026336, 0.0026336
3: 0.0012505, 0.0030566, 0.0012505, 0.0030566, -0.0011098, 0.0011098
4: 1.0016017, 1.0086088, 1.0016017, 1.0086088, -0.0043056, 0.0043056
5: 0.0025530, 0.0039162, 0.0025530, 0.0039162, -0.0008376, 0.0008376
6: -0.0108393, -0.0090653, -0.0108393, -0.0090653, -0.0010900, 0.0010900
7: -0.0101860, -0.0099597, -0.0101860, -0.0099597, -0.0001390, 0.0001390
8: -0.0045907, -0.0033650, -0.0045907, -0.0033650, -0.0007531, 0.0007531
9: -0.0013248, 0.0048113, -0.0013248, 0.0048113, -0.0037703, 0.0037703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018313, upper bound: 0.0017036
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0015491, upper bound: 0.0015491
time: 0.70 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 7.69 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024239, upper bound: 0.0024064
IS_A1_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024239, upper bound: 0.0024925
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024869, upper bound: 0.0024064
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024869, upper bound: 0.0025322
IS_A1_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0021930, upper bound: 0.0021827
IS_A1_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0020871, upper bound: 0.0021509
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0025505, upper bound: 0.0025036
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0025505, upper bound: 0.0025563
IS_A1_B1_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0019857, upper bound: 0.0016709
IS_A1_B1_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0016292, upper bound: 0.0015215
IS_A1_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0019857, upper bound: 0.0019108
IS_A1_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0016292, upper bound: 0.0017547
IS_A1_B1_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0020275, upper bound: 0.0017145
IS_A1_B1_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018101, upper bound: 0.0015976
IS_A1_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0020275, upper bound: 0.0021085
IS_A1_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018101, upper bound: 0.0020682
IS_A1_B1_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018682, upper bound: 0.0016457
IS_A1_B1_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0014101, upper bound: 0.0014101
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018682, upper bound: 0.0016457
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0014101, upper bound: 0.0014101
IS_A1_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018969, upper bound: 0.0016881
IS_A1_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0016170, upper bound: 0.0015372
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018969, upper bound: 0.0016881
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0016170, upper bound: 0.0015372
IS_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0020155, upper bound: 0.0018977
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015372, upper bound: 0.0016170
IS_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0020155, upper bound: 0.0018977
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015372, upper bound: 0.0016170
IS_A1_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0021769, upper bound: 0.0021091
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0020575, upper bound: 0.0020550
IS_A1_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0021769, upper bound: 0.0021091
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0020575, upper bound: 0.0020550
IS_A1_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018479, upper bound: 0.0017196
IS_A1_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015384, upper bound: 0.0015956
IS_A1_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024925, upper bound: 0.0025469
IS_A1_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024925, upper bound: 0.0026479
IS_A1_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015208, upper bound: 0.0018188
IS_A1_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0023337, upper bound: 0.0023888
IS_A1_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015351, upper bound: 0.0016133
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0023271, upper bound: 0.0024141
IS_A1_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015351, upper bound: 0.0016133
IS_A1_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0023271, upper bound: 0.0024141
IS_A1_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0025581, upper bound: 0.0024605
IS_A1_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0025581, upper bound: 0.0024621
IS_A1_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0025648, upper bound: 0.0024654
IS_A1_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0025648, upper bound: 0.0025075
IS_A1_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0023739, upper bound: 0.0022322
IS_A1_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0023739, upper bound: 0.0023005
IS_A1_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018138, upper bound: 0.0016339
IS_A1_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0013867, upper bound: 0.0014932
IS_A1_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0026135, upper bound: 0.0024464
IS_A1_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0026135, upper bound: 0.0025075
IS_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0017526, upper bound: 0.0018133
IS_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0014538, upper bound: 0.0016623
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0024036
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0024832
IS_A2_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018930, upper bound: 0.0018913
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0016351, upper bound: 0.0017355
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018930, upper bound: 0.0018913
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0016351, upper bound: 0.0017355
IS_A2_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024275, upper bound: 0.0023834
IS_A2_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024275, upper bound: 0.0024341
IS_A2_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0023834
IS_A2_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0024685, upper bound: 0.0024601
IS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018313, upper bound: 0.0017036
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015491, upper bound: 0.0015491
IS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0018313, upper bound: 0.0017036
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 7.69
Output dim: 4, lower bound: -0.0015491, upper bound: 0.0015491

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040229, -0.0037140, -0.0039958, -0.0036956, -0.0002029, 0.0001681
1: 0.0003123, 0.0020226, 0.0002104, 0.0018729, -0.0009309, 0.0011233
2: 0.0104475, 0.0142684, 0.0107818, 0.0144961, -0.0025096, 0.0020797
3: 0.0013216, 0.0029318, 0.0012257, 0.0027909, -0.0008764, 0.0010576
4: 1.0018777, 1.0081245, 1.0015054, 1.0075778, -0.0034001, 0.0041029
5: 0.0026067, 0.0038219, 0.0025343, 0.0037156, -0.0006615, 0.0007982
6: -0.0107166, -0.0091352, -0.0105783, -0.0090409, -0.0010387, 0.0008608
7: -0.0101704, -0.0099686, -0.0101527, -0.0099566, -0.0001325, 0.0001098
8: -0.0045424, -0.0034498, -0.0046075, -0.0035454, -0.0005947, 0.0007177
9: -0.0009005, 0.0045696, -0.0004219, 0.0048956, -0.0035928, 0.0029774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024131, upper bound: 0.0024847
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024131, upper bound: 0.0024925
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040120, -0.0037028, -0.0040090, -0.0037063, -0.0001860, 0.0001874
1: 0.0002503, 0.0019624, 0.0002696, 0.0019460, -0.0010376, 0.0010301
2: 0.0105819, 0.0144070, 0.0106186, 0.0143639, -0.0023013, 0.0023181
3: 0.0012632, 0.0028751, 0.0012814, 0.0028597, -0.0009769, 0.0009698
4: 1.0016510, 1.0079048, 1.0017216, 1.0078447, -0.0037899, 0.0037623
5: 0.0025626, 0.0037792, 0.0025763, 0.0037675, -0.0007373, 0.0007319
6: -0.0106610, -0.0090778, -0.0106458, -0.0090956, -0.0009525, 0.0009595
7: -0.0101633, -0.0099613, -0.0101613, -0.0099636, -0.0001215, 0.0001224
8: -0.0045821, -0.0034882, -0.0045697, -0.0034987, -0.0006629, 0.0006581
9: -0.0007081, 0.0047681, -0.0006555, 0.0047063, -0.0032946, 0.0033187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024131, upper bound: 0.0024039
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024131, upper bound: 0.0024064
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040229, -0.0037140, -0.0040090, -0.0037063, -0.0001859, 0.0001682
1: 0.0003123, 0.0020226, 0.0002696, 0.0019460, -0.0009315, 0.0010294
2: 0.0104475, 0.0142684, 0.0106186, 0.0143639, -0.0022997, 0.0020811
3: 0.0013216, 0.0029318, 0.0012814, 0.0028597, -0.0008770, 0.0009691
4: 1.0018777, 1.0081245, 1.0017216, 1.0078447, -0.0034023, 0.0037598
5: 0.0026067, 0.0038219, 0.0025763, 0.0037675, -0.0006619, 0.0007314
6: -0.0107166, -0.0091352, -0.0106458, -0.0090956, -0.0009518, 0.0008614
7: -0.0101704, -0.0099686, -0.0101613, -0.0099636, -0.0001214, 0.0001099
8: -0.0045424, -0.0034498, -0.0045697, -0.0034987, -0.0005951, 0.0006576
9: -0.0009005, 0.0045696, -0.0006555, 0.0047063, -0.0032924, 0.0029793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024131, upper bound: 0.0025305
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024131, upper bound: 0.0025305
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039959, -0.0036967, -0.0040091, -0.0037007, -0.0001895, 0.0001893
1: 0.0002167, 0.0018730, 0.0002390, 0.0019464, -0.0010483, 0.0010493
2: 0.0107816, 0.0144819, 0.0106177, 0.0144323, -0.0023442, 0.0023421
3: 0.0012316, 0.0027909, 0.0012525, 0.0028600, -0.0009870, 0.0009878
4: 1.0015286, 1.0075781, 1.0016097, 1.0078462, -0.0038291, 0.0038324
5: 0.0025388, 0.0037156, 0.0025545, 0.0037678, -0.0007449, 0.0007456
6: -0.0105783, -0.0090468, -0.0106462, -0.0090673, -0.0009702, 0.0009694
7: -0.0101527, -0.0099574, -0.0101614, -0.0099600, -0.0001238, 0.0001237
8: -0.0046035, -0.0035453, -0.0045893, -0.0034984, -0.0006698, 0.0006704
9: -0.0004221, 0.0048753, -0.0006568, 0.0048043, -0.0033560, 0.0033530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024063, upper bound: 0.0024239
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024063, upper bound: 0.0025036
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040091, -0.0037067, -0.0040091, -0.0037007, -0.0001890, 0.0001727
1: 0.0002719, 0.0019461, 0.0002390, 0.0019464, -0.0009563, 0.0010462
2: 0.0106184, 0.0143588, 0.0106177, 0.0144323, -0.0023374, 0.0021364
3: 0.0012835, 0.0028597, 0.0012525, 0.0028600, -0.0009003, 0.0009850
4: 1.0017298, 1.0078449, 1.0016097, 1.0078462, -0.0034928, 0.0038214
5: 0.0025779, 0.0037675, 0.0025545, 0.0037678, -0.0006795, 0.0007434
6: -0.0106459, -0.0090978, -0.0106462, -0.0090673, -0.0009674, 0.0008843
7: -0.0101613, -0.0099639, -0.0101614, -0.0099600, -0.0001234, 0.0001128
8: -0.0045683, -0.0034987, -0.0045893, -0.0034984, -0.0006109, 0.0006684
9: -0.0006558, 0.0046990, -0.0006568, 0.0048043, -0.0033463, 0.0030586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024063, upper bound: 0.0025305
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024063, upper bound: 0.0025561
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039959, -0.0036967, -0.0040544, -0.0037074, -0.0001912, 0.0002454
1: 0.0002167, 0.0018730, 0.0002757, 0.0021970, -0.0013585, 0.0010587
2: 0.0107816, 0.0144819, 0.0100578, 0.0143502, -0.0023653, 0.0030350
3: 0.0012316, 0.0027909, 0.0012871, 0.0030960, -0.0012790, 0.0009968
4: 1.0015286, 1.0075781, 1.0017439, 1.0087614, -0.0049619, 0.0038670
5: 0.0025388, 0.0037156, 0.0025807, 0.0039459, -0.0009653, 0.0007523
6: -0.0105783, -0.0090468, -0.0108779, -0.0091013, -0.0009790, 0.0012562
7: -0.0101527, -0.0099574, -0.0101909, -0.0099643, -0.0001249, 0.0001602
8: -0.0046035, -0.0035453, -0.0045658, -0.0033383, -0.0008679, 0.0006764
9: -0.0004221, 0.0048753, -0.0014583, 0.0046867, -0.0033862, 0.0043450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0022824, upper bound: 0.0023487
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022824, upper bound: 0.0025469
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040091, -0.0037067, -0.0040544, -0.0037074, -0.0001934, 0.0002303
1: 0.0002719, 0.0019461, 0.0002757, 0.0021970, -0.0012752, 0.0010711
2: 0.0106184, 0.0143588, 0.0100578, 0.0143502, -0.0023930, 0.0028488
3: 0.0012835, 0.0028597, 0.0012871, 0.0030960, -0.0012005, 0.0010084
4: 1.0017298, 1.0078449, 1.0017439, 1.0087614, -0.0046575, 0.0039122
5: 0.0025779, 0.0037675, 0.0025807, 0.0039459, -0.0009061, 0.0007611
6: -0.0106459, -0.0090978, -0.0108779, -0.0091013, -0.0009904, 0.0011791
7: -0.0101613, -0.0099639, -0.0101909, -0.0099643, -0.0001263, 0.0001504
8: -0.0045683, -0.0034987, -0.0045658, -0.0033383, -0.0008147, 0.0006843
9: -0.0006558, 0.0046990, -0.0014583, 0.0046867, -0.0034258, 0.0040785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022824, upper bound: 0.0025648
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0022824, upper bound: 0.0026479
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040673, -0.0037214, -0.0040265, -0.0037136, -0.0002327, 0.0001974
1: 0.0003532, 0.0022688, 0.0003100, 0.0020427, -0.0010929, 0.0012883
2: 0.0098973, 0.0141770, 0.0104026, 0.0142737, -0.0028782, 0.0024417
3: 0.0013601, 0.0031636, 0.0013194, 0.0029507, -0.0010289, 0.0012129
4: 1.0020272, 1.0090239, 1.0018691, 1.0081978, -0.0039918, 0.0047055
5: 0.0026358, 0.0039969, 0.0026050, 0.0038362, -0.0007766, 0.0009154
6: -0.0109443, -0.0091730, -0.0107352, -0.0091330, -0.0011913, 0.0010106
7: -0.0101994, -0.0099735, -0.0101727, -0.0099684, -0.0001520, 0.0001289
8: -0.0045163, -0.0032924, -0.0045439, -0.0034370, -0.0006982, 0.0008231
9: -0.0016881, 0.0044388, -0.0009647, 0.0045772, -0.0041205, 0.0034955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.63 + 596.56 = 600.20 seconds
