## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00017731


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0017310, -0.0013944, -0.0017310, -0.0013944, -0.0001731, 0.0001731)
1: (-0.0087033, -0.0078491, -0.0087033, -0.0078491, -0.0004392, 0.0004392)
2: (0.0296304, 0.0301604, 0.0296304, 0.0301604, -0.0002725, 0.0002725)
3: (0.0032454, 0.0042350, 0.0032454, 0.0042350, -0.0005088, 0.0005088)
4: (-0.0077458, -0.0068769, -0.0077458, -0.0068769, -0.0004467, 0.0004467)
5: (0.0108043, 0.0111334, 0.0108043, 0.0111334, -0.0001692, 0.0001692)
6: (0.0045070, 0.0057628, 0.0045070, 0.0057628, -0.0006457, 0.0006457)
7: (0.9812130, 0.9820918, 0.9812130, 0.9820918, -0.0004518, 0.0004518)
8: (-0.0067068, -0.0057646, -0.0067068, -0.0057646, -0.0004844, 0.0004844)
9: (-0.0011917, -0.0005694, -0.0011917, -0.0005694, -0.0003200, 0.0003200)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 1.34 = 2.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0002598, upper bound: 0.0002597

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002431, upper bound: 0.0002398
time: 0.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002443, upper bound: 0.0002442
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 7, lower bound: -0.0002431, upper bound: 0.0002398
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 7, lower bound: -0.0002443, upper bound: 0.0002442

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013819, -0.0017206, -0.0013947, -0.0001321, 0.0001573
1: -0.0086337, -0.0078174, -0.0086769, -0.0078497, -0.0003351, 0.0003991
2: 0.0296737, 0.0301801, 0.0296468, 0.0301600, -0.0002079, 0.0002476
3: 0.0032087, 0.0041543, 0.0032461, 0.0042044, -0.0004623, 0.0003882
4: -0.0076749, -0.0068447, -0.0077189, -0.0068775, -0.0003409, 0.0004059
5: 0.0108311, 0.0111456, 0.0108145, 0.0111332, -0.0001291, 0.0001538
6: 0.0044604, 0.0056604, 0.0045079, 0.0057240, -0.0005867, 0.0004927
7: 0.9811804, 0.9820202, 0.9812137, 0.9820646, -0.0004106, 0.0003448
8: -0.0067418, -0.0058414, -0.0067062, -0.0057937, -0.0004402, 0.0003697
9: -0.0011410, -0.0005463, -0.0011725, -0.0005698, -0.0002442, 0.0002908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002286, upper bound: 0.0002195
time: 0.57 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002239, upper bound: 0.0002195
time: 0.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0013946, -0.0017309, -0.0013944, -0.0001360, 0.0001726
1: -0.0086901, -0.0078495, -0.0087031, -0.0078491, -0.0003450, 0.0004379
2: 0.0296387, 0.0301602, 0.0296306, 0.0301604, -0.0002141, 0.0002717
3: 0.0032458, 0.0042196, 0.0032454, 0.0042347, -0.0005073, 0.0003997
4: -0.0077323, -0.0068773, -0.0077455, -0.0068769, -0.0003509, 0.0004455
5: 0.0108094, 0.0111333, 0.0108044, 0.0111334, -0.0001329, 0.0001687
6: 0.0045075, 0.0057434, 0.0045070, 0.0057625, -0.0006439, 0.0005073
7: 0.9812134, 0.9820782, 0.9812130, 0.9820916, -0.0004505, 0.0003550
8: -0.0067064, -0.0057792, -0.0067068, -0.0057649, -0.0004831, 0.0003806
9: -0.0011821, -0.0005696, -0.0011916, -0.0005694, -0.0002514, 0.0003191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002341, upper bound: 0.0002298
time: 0.55 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002301, upper bound: 0.0002301
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.50 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 7, lower bound: -0.0002286, upper bound: 0.0002195
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 7, lower bound: -0.0002239, upper bound: 0.0002195
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 7, lower bound: -0.0002341, upper bound: 0.0002298
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 7, lower bound: -0.0002301, upper bound: 0.0002301

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013888, -0.0017206, -0.0013960, -0.0001297, 0.0001457
1: -0.0086337, -0.0078350, -0.0086769, -0.0078532, -0.0003292, 0.0003697
2: 0.0296737, 0.0301692, 0.0296468, 0.0301579, -0.0002042, 0.0002294
3: 0.0032290, 0.0041543, 0.0032501, 0.0042044, -0.0004283, 0.0003814
4: -0.0076749, -0.0068625, -0.0077189, -0.0068810, -0.0003349, 0.0003761
5: 0.0108311, 0.0111388, 0.0108145, 0.0111318, -0.0001268, 0.0001424
6: 0.0044862, 0.0056604, 0.0045130, 0.0057240, -0.0005436, 0.0004840
7: 0.9811985, 0.9820201, 0.9812172, 0.9820646, -0.0003804, 0.0003387
8: -0.0067224, -0.0058415, -0.0067023, -0.0057937, -0.0004078, 0.0003631
9: -0.0011410, -0.0005591, -0.0011725, -0.0005723, -0.0002399, 0.0002694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002036, upper bound: 0.0001827
time: 0.51 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002036, upper bound: 0.0001782
time: 0.52 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0017114, -0.0013973, -0.0017206, -0.0014021, -0.0001504, 0.0001497
1: -0.0086535, -0.0078563, -0.0086769, -0.0078687, -0.0003816, 0.0003799
2: 0.0296614, 0.0301559, 0.0296468, 0.0301483, -0.0002367, 0.0002357
3: 0.0032538, 0.0041773, 0.0032681, 0.0042044, -0.0004401, 0.0004421
4: -0.0076951, -0.0068842, -0.0077189, -0.0068968, -0.0003881, 0.0003864
5: 0.0108235, 0.0111306, 0.0108145, 0.0111259, -0.0001470, 0.0001464
6: 0.0045176, 0.0056896, 0.0045357, 0.0057240, -0.0005586, 0.0005610
7: 0.9812204, 0.9820406, 0.9812331, 0.9820647, -0.0003908, 0.0003926
8: -0.0066989, -0.0058196, -0.0066852, -0.0057937, -0.0004191, 0.0004209
9: -0.0011554, -0.0005746, -0.0011725, -0.0005836, -0.0002780, 0.0002768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002044, upper bound: 0.0001791
time: 0.53 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001936, upper bound: 0.0001762
time: 0.53 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014006, -0.0017309, -0.0013958, -0.0001327, 0.0001647
1: -0.0086901, -0.0078648, -0.0087031, -0.0078526, -0.0003366, 0.0004180
2: 0.0296387, 0.0301506, 0.0296306, 0.0301583, -0.0002088, 0.0002593
3: 0.0032636, 0.0042196, 0.0032494, 0.0042347, -0.0004842, 0.0003900
4: -0.0077323, -0.0068929, -0.0077455, -0.0068804, -0.0003424, 0.0004252
5: 0.0108094, 0.0111273, 0.0108044, 0.0111321, -0.0001297, 0.0001610
6: 0.0045301, 0.0057433, 0.0045120, 0.0057625, -0.0006145, 0.0004949
7: 0.9812292, 0.9820782, 0.9812165, 0.9820916, -0.0004300, 0.0003463
8: -0.0066895, -0.0057792, -0.0067030, -0.0057649, -0.0004611, 0.0003713
9: -0.0011821, -0.0005808, -0.0011916, -0.0005719, -0.0002453, 0.0003046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002233, upper bound: 0.0002238
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002233, upper bound: 0.0002299
time: 0.52 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014103, -0.0017309, -0.0014019, -0.0001576, 0.0001681
1: -0.0087082, -0.0078893, -0.0087031, -0.0078680, -0.0003999, 0.0004265
2: 0.0296274, 0.0301355, 0.0296306, 0.0301486, -0.0002481, 0.0002646
3: 0.0032920, 0.0042406, 0.0032673, 0.0042347, -0.0004941, 0.0004633
4: -0.0077507, -0.0069178, -0.0077455, -0.0068962, -0.0004068, 0.0004338
5: 0.0108024, 0.0111179, 0.0108044, 0.0111261, -0.0001541, 0.0001643
6: 0.0045661, 0.0057700, 0.0045348, 0.0057625, -0.0006270, 0.0005880
7: 0.9812544, 0.9820968, 0.9812325, 0.9820916, -0.0004388, 0.0004115
8: -0.0066625, -0.0057593, -0.0066859, -0.0057649, -0.0004704, 0.0004412
9: -0.0011953, -0.0005987, -0.0011916, -0.0005832, -0.0002914, 0.0003107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002195, upper bound: 0.0002239
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002195, upper bound: 0.0002301
time: 0.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.46 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0002036, upper bound: 0.0001827
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0002036, upper bound: 0.0001782
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0002044, upper bound: 0.0001791
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0001936, upper bound: 0.0001762
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0002233, upper bound: 0.0002238
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0002233, upper bound: 0.0002299
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0002195, upper bound: 0.0002239
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 7, lower bound: -0.0002195, upper bound: 0.0002301

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013897, -0.0017206, -0.0014004, -0.0001242, 0.0001435
1: -0.0086336, -0.0078373, -0.0086769, -0.0078644, -0.0003153, 0.0003641
2: 0.0296737, 0.0301678, 0.0296468, 0.0301509, -0.0001956, 0.0002259
3: 0.0032317, 0.0041543, 0.0032631, 0.0042044, -0.0004218, 0.0003653
4: -0.0076749, -0.0068648, -0.0077189, -0.0068924, -0.0003207, 0.0003703
5: 0.0108311, 0.0111380, 0.0108145, 0.0111275, -0.0001215, 0.0001403
6: 0.0044895, 0.0056604, 0.0045294, 0.0057240, -0.0005353, 0.0004635
7: 0.9812009, 0.9820201, 0.9812287, 0.9820647, -0.0003746, 0.0003244
8: -0.0067199, -0.0058415, -0.0066900, -0.0057938, -0.0004016, 0.0003478
9: -0.0011410, -0.0005607, -0.0011725, -0.0005805, -0.0002297, 0.0002653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001695
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001764
time: 0.56 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013948, -0.0017277, -0.0014106, -0.0001242, 0.0001548
1: -0.0086336, -0.0078500, -0.0086948, -0.0078902, -0.0003152, 0.0003929
2: 0.0296737, 0.0301599, 0.0296357, 0.0301349, -0.0001956, 0.0002438
3: 0.0032464, 0.0041542, 0.0032930, 0.0042251, -0.0004552, 0.0003652
4: -0.0076749, -0.0068778, -0.0077371, -0.0069187, -0.0003207, 0.0003996
5: 0.0108311, 0.0111331, 0.0108076, 0.0111176, -0.0001215, 0.0001514
6: 0.0045082, 0.0056604, 0.0045674, 0.0057503, -0.0005776, 0.0004635
7: 0.9812139, 0.9820201, 0.9812553, 0.9820830, -0.0004042, 0.0003243
8: -0.0067059, -0.0058415, -0.0066615, -0.0057740, -0.0004334, 0.0003477
9: -0.0011410, -0.0005700, -0.0011855, -0.0005993, -0.0002297, 0.0002863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001656
time: 0.57 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001710
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0017114, -0.0014025, -0.0017206, -0.0014029, -0.0001486, 0.0001418
1: -0.0086535, -0.0078695, -0.0086769, -0.0078707, -0.0003771, 0.0003598
2: 0.0296614, 0.0301477, 0.0296468, 0.0301470, -0.0002340, 0.0002232
3: 0.0032690, 0.0041772, 0.0032704, 0.0042044, -0.0004168, 0.0004369
4: -0.0076951, -0.0068976, -0.0077189, -0.0068989, -0.0003836, 0.0003660
5: 0.0108235, 0.0111255, 0.0108145, 0.0111251, -0.0001453, 0.0001386
6: 0.0045369, 0.0056896, 0.0045387, 0.0057240, -0.0005290, 0.0005544
7: 0.9812340, 0.9820405, 0.9812352, 0.9820647, -0.0003702, 0.0003880
8: -0.0066843, -0.0058196, -0.0066830, -0.0057937, -0.0003969, 0.0004160
9: -0.0011554, -0.0005842, -0.0011725, -0.0005851, -0.0002748, 0.0002622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001987, upper bound: 0.0001681
time: 0.58 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001987, upper bound: 0.0001727
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0017189, -0.0014114, -0.0017206, -0.0014085, -0.0001577, 0.0001417
1: -0.0086725, -0.0078922, -0.0086769, -0.0078850, -0.0004002, 0.0003595
2: 0.0296496, 0.0301336, 0.0296468, 0.0301382, -0.0002483, 0.0002231
3: 0.0032954, 0.0041993, 0.0032869, 0.0042044, -0.0004165, 0.0004637
4: -0.0077144, -0.0069207, -0.0077189, -0.0069133, -0.0004071, 0.0003657
5: 0.0108162, 0.0111168, 0.0108145, 0.0111196, -0.0001542, 0.0001385
6: 0.0045703, 0.0057175, 0.0045597, 0.0057240, -0.0005286, 0.0005884
7: 0.9812573, 0.9820601, 0.9812500, 0.9820647, -0.0003699, 0.0004118
8: -0.0066593, -0.0057986, -0.0066673, -0.0057938, -0.0003966, 0.0004415
9: -0.0011693, -0.0006008, -0.0011725, -0.0005955, -0.0002916, 0.0002620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001866, upper bound: 0.0001627
time: 0.58 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001866, upper bound: 0.0001685
time: 0.54 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014006, -0.0017036, -0.0013834, -0.0001693, 0.0001241
1: -0.0086901, -0.0078648, -0.0086337, -0.0078211, -0.0004297, 0.0003149
2: 0.0296387, 0.0301506, 0.0296737, 0.0301778, -0.0002666, 0.0001954
3: 0.0032636, 0.0042196, 0.0032129, 0.0041543, -0.0003648, 0.0004978
4: -0.0077323, -0.0068929, -0.0076749, -0.0068484, -0.0004371, 0.0003203
5: 0.0108094, 0.0111273, 0.0108311, 0.0111442, -0.0001656, 0.0001213
6: 0.0045301, 0.0057433, 0.0044657, 0.0056604, -0.0004630, 0.0006318
7: 0.9812292, 0.9820782, 0.9811842, 0.9820202, -0.0003240, 0.0004421
8: -0.0066895, -0.0057792, -0.0067378, -0.0058414, -0.0003474, 0.0004740
9: -0.0011821, -0.0005808, -0.0011410, -0.0005489, -0.0003131, 0.0002294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001916, upper bound: 0.0001936
time: 0.52 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001852, upper bound: 0.0001936
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014006, -0.0017258, -0.0013959, -0.0001323, 0.0001242
1: -0.0086901, -0.0078648, -0.0086901, -0.0078529, -0.0003358, 0.0003152
2: 0.0296387, 0.0301506, 0.0296387, 0.0301580, -0.0002083, 0.0001956
3: 0.0032636, 0.0042196, 0.0032499, 0.0042196, -0.0003652, 0.0003890
4: -0.0077323, -0.0068929, -0.0077323, -0.0068808, -0.0003415, 0.0003206
5: 0.0108094, 0.0111273, 0.0108094, 0.0111319, -0.0001294, 0.0001214
6: 0.0045301, 0.0057433, 0.0045126, 0.0057434, -0.0004634, 0.0004937
7: 0.9812292, 0.9820782, 0.9812170, 0.9820782, -0.0003243, 0.0003454
8: -0.0066895, -0.0057792, -0.0067026, -0.0057792, -0.0003477, 0.0003704
9: -0.0011821, -0.0005808, -0.0011821, -0.0005722, -0.0002447, 0.0002297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001916, upper bound: 0.0002178
time: 0.52 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001852, upper bound: 0.0002178
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014103, -0.0017036, -0.0013895, -0.0001833, 0.0001273
1: -0.0087082, -0.0078893, -0.0086337, -0.0078366, -0.0004651, 0.0003231
2: 0.0296274, 0.0301355, 0.0296737, 0.0301682, -0.0002886, 0.0002004
3: 0.0032920, 0.0042406, 0.0032309, 0.0041543, -0.0003743, 0.0005388
4: -0.0077507, -0.0069178, -0.0076749, -0.0068641, -0.0004731, 0.0003286
5: 0.0108024, 0.0111179, 0.0108311, 0.0111382, -0.0001792, 0.0001245
6: 0.0045661, 0.0057700, 0.0044885, 0.0056604, -0.0004750, 0.0006838
7: 0.9812544, 0.9820968, 0.9812001, 0.9820201, -0.0003324, 0.0004785
8: -0.0066625, -0.0057593, -0.0067207, -0.0058415, -0.0003564, 0.0005130
9: -0.0011953, -0.0005987, -0.0011410, -0.0005602, -0.0003389, 0.0002354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001817, upper bound: 0.0001936
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001762, upper bound: 0.0001936
time: 0.55 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014103, -0.0017258, -0.0014020, -0.0001573, 0.0001281
1: -0.0087082, -0.0078893, -0.0086901, -0.0078684, -0.0003991, 0.0003252
2: 0.0296274, 0.0301355, 0.0296387, 0.0301484, -0.0002476, 0.0002017
3: 0.0032920, 0.0042406, 0.0032678, 0.0042196, -0.0003767, 0.0004624
4: -0.0077507, -0.0069178, -0.0077323, -0.0068965, -0.0004060, 0.0003307
5: 0.0108024, 0.0111179, 0.0108094, 0.0111260, -0.0001538, 0.0001253
6: 0.0045661, 0.0057700, 0.0045354, 0.0057433, -0.0004780, 0.0005868
7: 0.9812544, 0.9820968, 0.9812328, 0.9820782, -0.0003345, 0.0004106
8: -0.0066625, -0.0057593, -0.0066855, -0.0057793, -0.0003587, 0.0004403
9: -0.0011953, -0.0005987, -0.0011821, -0.0005834, -0.0002908, 0.0002369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001817, upper bound: 0.0002180
time: 0.51 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001762, upper bound: 0.0002180
time: 0.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.53 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001695
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001764
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001656
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001710
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001987, upper bound: 0.0001681
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001987, upper bound: 0.0001727
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001866, upper bound: 0.0001627
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001866, upper bound: 0.0001685
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001916, upper bound: 0.0001936
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001852, upper bound: 0.0001936
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001916, upper bound: 0.0002178
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001852, upper bound: 0.0002178
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001817, upper bound: 0.0001936
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001762, upper bound: 0.0001936
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001817, upper bound: 0.0002180
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 7, lower bound: -0.0001762, upper bound: 0.0002180

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013930, -0.0017206, -0.0014090, -0.0001148, 0.0001379
1: -0.0086336, -0.0078454, -0.0086768, -0.0078860, -0.0002913, 0.0003499
2: 0.0296737, 0.0301627, 0.0296469, 0.0301375, -0.0001807, 0.0002171
3: 0.0032411, 0.0041542, 0.0032881, 0.0042042, -0.0004054, 0.0003374
4: -0.0076749, -0.0068731, -0.0077188, -0.0069144, -0.0002963, 0.0003559
5: 0.0108312, 0.0111348, 0.0108145, 0.0111192, -0.0001122, 0.0001348
6: 0.0045015, 0.0056603, 0.0045612, 0.0057238, -0.0005145, 0.0004282
7: 0.9812092, 0.9820201, 0.9812510, 0.9820645, -0.0003600, 0.0002997
8: -0.0067109, -0.0058415, -0.0066661, -0.0057939, -0.0003860, 0.0003213
9: -0.0011409, -0.0005667, -0.0011724, -0.0005962, -0.0002122, 0.0002550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001695
time: 0.54 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001695
time: 0.54 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013918, -0.0017233, -0.0014053, -0.0001172, 0.0001423
1: -0.0086336, -0.0078425, -0.0086836, -0.0078767, -0.0002973, 0.0003611
2: 0.0296737, 0.0301645, 0.0296427, 0.0301433, -0.0001844, 0.0002240
3: 0.0032377, 0.0041542, 0.0032773, 0.0042121, -0.0004183, 0.0003444
4: -0.0076749, -0.0068701, -0.0077257, -0.0069049, -0.0003024, 0.0003673
5: 0.0108311, 0.0111360, 0.0108119, 0.0111228, -0.0001145, 0.0001391
6: 0.0044972, 0.0056604, 0.0045475, 0.0057338, -0.0005308, 0.0004371
7: 0.9812063, 0.9820201, 0.9812414, 0.9820715, -0.0003715, 0.0003059
8: -0.0067142, -0.0058415, -0.0066764, -0.0057864, -0.0003983, 0.0003279
9: -0.0011410, -0.0005645, -0.0011774, -0.0005894, -0.0002166, 0.0002631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001764
time: 0.54 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001764
time: 0.55 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013979, -0.0017276, -0.0014186, -0.0001147, 0.0001495
1: -0.0086336, -0.0078581, -0.0086947, -0.0079104, -0.0002911, 0.0003793
2: 0.0296737, 0.0301549, 0.0296358, 0.0301224, -0.0001806, 0.0002353
3: 0.0032558, 0.0041542, 0.0033164, 0.0042249, -0.0004395, 0.0003373
4: -0.0076748, -0.0068860, -0.0077370, -0.0069393, -0.0002961, 0.0003859
5: 0.0108312, 0.0111300, 0.0108076, 0.0111098, -0.0001122, 0.0001462
6: 0.0045201, 0.0056603, 0.0045971, 0.0057501, -0.0005577, 0.0004280
7: 0.9812222, 0.9820201, 0.9812762, 0.9820829, -0.0003903, 0.0002995
8: -0.0066970, -0.0058415, -0.0066392, -0.0057742, -0.0004184, 0.0003211
9: -0.0011409, -0.0005759, -0.0011854, -0.0006140, -0.0002121, 0.0002764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001656
time: 0.57 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001656
time: 0.58 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013969, -0.0017302, -0.0014158, -0.0001171, 0.0001509
1: -0.0086336, -0.0078553, -0.0087012, -0.0079034, -0.0002972, 0.0003830
2: 0.0296737, 0.0301565, 0.0296317, 0.0301267, -0.0001844, 0.0002376
3: 0.0032526, 0.0041542, 0.0033083, 0.0042325, -0.0004437, 0.0003443
4: -0.0076749, -0.0068832, -0.0077436, -0.0069321, -0.0003023, 0.0003896
5: 0.0108312, 0.0111310, 0.0108051, 0.0111125, -0.0001145, 0.0001476
6: 0.0045161, 0.0056603, 0.0045868, 0.0057598, -0.0005631, 0.0004370
7: 0.9812195, 0.9820201, 0.9812689, 0.9820897, -0.0003940, 0.0003058
8: -0.0067000, -0.0058415, -0.0066469, -0.0057669, -0.0004225, 0.0003278
9: -0.0011409, -0.0005739, -0.0011902, -0.0006089, -0.0002166, 0.0002791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001710
time: 0.54 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001710
time: 0.54 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017114, -0.0014056, -0.0017206, -0.0014111, -0.0001391, 0.0001366
1: -0.0086534, -0.0078775, -0.0086768, -0.0078915, -0.0003530, 0.0003466
2: 0.0296614, 0.0301428, 0.0296469, 0.0301341, -0.0002190, 0.0002150
3: 0.0032783, 0.0041772, 0.0032945, 0.0042042, -0.0004015, 0.0004089
4: -0.0076950, -0.0069058, -0.0077188, -0.0069200, -0.0003591, 0.0003525
5: 0.0108235, 0.0111225, 0.0108145, 0.0111171, -0.0001360, 0.0001335
6: 0.0045487, 0.0056895, 0.0045693, 0.0057238, -0.0005095, 0.0005190
7: 0.9812422, 0.9820405, 0.9812566, 0.9820645, -0.0003566, 0.0003632
8: -0.0066755, -0.0058197, -0.0066601, -0.0057939, -0.0003823, 0.0003894
9: -0.0011554, -0.0005900, -0.0011724, -0.0006003, -0.0002572, 0.0002525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001651
time: 0.55 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001681
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017114, -0.0014045, -0.0017233, -0.0014078, -0.0001412, 0.0001406
1: -0.0086534, -0.0078748, -0.0086836, -0.0078832, -0.0003582, 0.0003567
2: 0.0296614, 0.0301445, 0.0296427, 0.0301393, -0.0002223, 0.0002213
3: 0.0032751, 0.0041772, 0.0032849, 0.0042121, -0.0004133, 0.0004150
4: -0.0076950, -0.0069030, -0.0077257, -0.0069116, -0.0003644, 0.0003629
5: 0.0108235, 0.0111235, 0.0108119, 0.0111203, -0.0001380, 0.0001374
6: 0.0045447, 0.0056895, 0.0045571, 0.0057338, -0.0005245, 0.0005267
7: 0.9812394, 0.9820405, 0.9812481, 0.9820715, -0.0003670, 0.0003686
8: -0.0066785, -0.0058196, -0.0066692, -0.0057864, -0.0003935, 0.0003952
9: -0.0011554, -0.0005881, -0.0011774, -0.0005942, -0.0002610, 0.0002599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001678
time: 0.57 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001727
time: 0.56 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017189, -0.0014145, -0.0017206, -0.0014166, -0.0001484, 0.0001364
1: -0.0086725, -0.0079001, -0.0086768, -0.0079053, -0.0003765, 0.0003462
2: 0.0296496, 0.0301288, 0.0296469, 0.0301255, -0.0002336, 0.0002148
3: 0.0033045, 0.0041992, 0.0033105, 0.0042042, -0.0004011, 0.0004361
4: -0.0077144, -0.0069288, -0.0077188, -0.0069341, -0.0003829, 0.0003522
5: 0.0108162, 0.0111137, 0.0108145, 0.0111117, -0.0001451, 0.0001334
6: 0.0045819, 0.0057175, 0.0045896, 0.0057238, -0.0005090, 0.0005535
7: 0.9812655, 0.9820601, 0.9812708, 0.9820645, -0.0003562, 0.0003873
8: -0.0066506, -0.0057987, -0.0066448, -0.0057939, -0.0003819, 0.0004153
9: -0.0011693, -0.0006065, -0.0011724, -0.0006103, -0.0002743, 0.0002523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001632, upper bound: 0.0001587
time: 0.58 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001632, upper bound: 0.0001627
time: 0.55 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017189, -0.0014135, -0.0017233, -0.0014135, -0.0001506, 0.0001406
1: -0.0086725, -0.0078976, -0.0086836, -0.0078975, -0.0003821, 0.0003569
2: 0.0296496, 0.0301303, 0.0296427, 0.0301304, -0.0002371, 0.0002214
3: 0.0033016, 0.0041992, 0.0033015, 0.0042121, -0.0004134, 0.0004426
4: -0.0077144, -0.0069262, -0.0077257, -0.0069261, -0.0003887, 0.0003630
5: 0.0108162, 0.0111147, 0.0108119, 0.0111147, -0.0001472, 0.0001375
6: 0.0045783, 0.0057175, 0.0045781, 0.0057338, -0.0005247, 0.0005618
7: 0.9812630, 0.9820601, 0.9812627, 0.9820715, -0.0003672, 0.0003931
8: -0.0066533, -0.0057986, -0.0066534, -0.0057864, -0.0003937, 0.0004215
9: -0.0011693, -0.0006047, -0.0011774, -0.0006046, -0.0002784, 0.0002600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001621
time: 0.55 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001685
time: 0.54 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014051, -0.0017036, -0.0013842, -0.0001669, 0.0001184
1: -0.0086900, -0.0078762, -0.0086337, -0.0078233, -0.0004236, 0.0003004
2: 0.0296387, 0.0301436, 0.0296737, 0.0301764, -0.0002628, 0.0001864
3: 0.0032768, 0.0042196, 0.0032155, 0.0041543, -0.0003480, 0.0004907
4: -0.0077322, -0.0069044, -0.0076749, -0.0068506, -0.0004308, 0.0003056
5: 0.0108094, 0.0111230, 0.0108311, 0.0111434, -0.0001632, 0.0001157
6: 0.0045468, 0.0057433, 0.0044690, 0.0056604, -0.0004417, 0.0006227
7: 0.9812409, 0.9820781, 0.9811865, 0.9820201, -0.0003091, 0.0004358
8: -0.0066770, -0.0057793, -0.0067353, -0.0058415, -0.0003314, 0.0004672
9: -0.0011821, -0.0005891, -0.0011410, -0.0005505, -0.0003086, 0.0002189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001831
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001866
time: 0.54 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0017317, -0.0014152, -0.0017036, -0.0013893, -0.0001755, 0.0001184
1: -0.0087051, -0.0079019, -0.0086336, -0.0078362, -0.0004454, 0.0003005
2: 0.0296294, 0.0301277, 0.0296737, 0.0301684, -0.0002764, 0.0001864
3: 0.0033065, 0.0042370, 0.0032304, 0.0041542, -0.0003481, 0.0005160
4: -0.0077475, -0.0069306, -0.0076749, -0.0068637, -0.0004531, 0.0003057
5: 0.0108036, 0.0111131, 0.0108311, 0.0111384, -0.0001716, 0.0001158
6: 0.0045846, 0.0057654, 0.0044879, 0.0056604, -0.0004418, 0.0006549
7: 0.9812673, 0.9820936, 0.9811997, 0.9820201, -0.0003092, 0.0004583
8: -0.0066486, -0.0057627, -0.0067211, -0.0058415, -0.0003315, 0.0004913
9: -0.0011930, -0.0006078, -0.0011410, -0.0005599, -0.0003246, 0.0002190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001826
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001866
time: 0.54 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014051, -0.0017258, -0.0013967, -0.0001303, 0.0001166
1: -0.0086900, -0.0078762, -0.0086901, -0.0078549, -0.0003305, 0.0002958
2: 0.0296387, 0.0301436, 0.0296387, 0.0301568, -0.0002051, 0.0001835
3: 0.0032768, 0.0042196, 0.0032521, 0.0042196, -0.0003427, 0.0003829
4: -0.0077322, -0.0069044, -0.0077323, -0.0068827, -0.0003362, 0.0003009
5: 0.0108094, 0.0111230, 0.0108094, 0.0111312, -0.0001273, 0.0001140
6: 0.0045468, 0.0057433, 0.0045154, 0.0057433, -0.0004349, 0.0004860
7: 0.9812409, 0.9820781, 0.9812189, 0.9820782, -0.0003043, 0.0003401
8: -0.0066770, -0.0057793, -0.0067005, -0.0057792, -0.0003263, 0.0003646
9: -0.0011821, -0.0005891, -0.0011821, -0.0005735, -0.0002408, 0.0002155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002080
time: 0.57 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002125
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0017317, -0.0014152, -0.0017258, -0.0014024, -0.0001475, 0.0001151
1: -0.0087051, -0.0079019, -0.0086900, -0.0078693, -0.0003742, 0.0002920
2: 0.0296294, 0.0301277, 0.0296387, 0.0301479, -0.0002322, 0.0001812
3: 0.0033065, 0.0042370, 0.0032688, 0.0042196, -0.0003383, 0.0004335
4: -0.0077475, -0.0069306, -0.0077323, -0.0068975, -0.0003806, 0.0002971
5: 0.0108036, 0.0111131, 0.0108094, 0.0111256, -0.0001442, 0.0001125
6: 0.0045846, 0.0057654, 0.0045367, 0.0057433, -0.0004294, 0.0005502
7: 0.9812673, 0.9820936, 0.9812338, 0.9820781, -0.0003005, 0.0003850
8: -0.0066486, -0.0057627, -0.0066845, -0.0057793, -0.0003221, 0.0004127
9: -0.0011930, -0.0006078, -0.0011821, -0.0005841, -0.0002726, 0.0002128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002147, upper bound: 0.0002080
time: 0.59 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002148, upper bound: 0.0002125
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014149, -0.0017036, -0.0013904, -0.0001810, 0.0001217
1: -0.0087081, -0.0079012, -0.0086336, -0.0078388, -0.0004592, 0.0003087
2: 0.0296275, 0.0301281, 0.0296737, 0.0301668, -0.0002849, 0.0001915
3: 0.0033057, 0.0042405, 0.0032335, 0.0041542, -0.0003576, 0.0005320
4: -0.0077507, -0.0069299, -0.0076749, -0.0068664, -0.0004671, 0.0003140
5: 0.0108024, 0.0111133, 0.0108311, 0.0111374, -0.0001769, 0.0001189
6: 0.0045835, 0.0057699, 0.0044918, 0.0056604, -0.0004539, 0.0006752
7: 0.9812666, 0.9820968, 0.9812025, 0.9820201, -0.0003176, 0.0004724
8: -0.0066494, -0.0057593, -0.0067182, -0.0058415, -0.0003405, 0.0005065
9: -0.0011953, -0.0006073, -0.0011410, -0.0005618, -0.0003346, 0.0002249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001831
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001866
time: 0.55 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0017388, -0.0014253, -0.0017036, -0.0013955, -0.0001889, 0.0001218
1: -0.0087231, -0.0079274, -0.0086336, -0.0078520, -0.0004792, 0.0003092
2: 0.0296182, 0.0301118, 0.0296737, 0.0301586, -0.0002973, 0.0001918
3: 0.0033362, 0.0042579, 0.0032487, 0.0041542, -0.0003582, 0.0005552
4: -0.0077659, -0.0069566, -0.0076749, -0.0068798, -0.0004875, 0.0003145
5: 0.0107967, 0.0111032, 0.0108311, 0.0111323, -0.0001846, 0.0001191
6: 0.0046221, 0.0057919, 0.0045112, 0.0056604, -0.0004546, 0.0007046
7: 0.9812936, 0.9821122, 0.9812160, 0.9820201, -0.0003181, 0.0004930
8: -0.0066204, -0.0057428, -0.0067037, -0.0058415, -0.0003411, 0.0005286
9: -0.0012062, -0.0006264, -0.0011410, -0.0005714, -0.0003492, 0.0002253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001691, upper bound: 0.0001826
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001866
time: 0.54 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014149, -0.0017258, -0.0014028, -0.0001552, 0.0001203
1: -0.0087081, -0.0079012, -0.0086900, -0.0078705, -0.0003937, 0.0003053
2: 0.0296275, 0.0301281, 0.0296387, 0.0301471, -0.0002443, 0.0001894
3: 0.0033057, 0.0042405, 0.0032701, 0.0042196, -0.0003537, 0.0004561
4: -0.0077507, -0.0069299, -0.0077323, -0.0068986, -0.0004005, 0.0003106
5: 0.0108024, 0.0111133, 0.0108094, 0.0111252, -0.0001517, 0.0001176
6: 0.0045835, 0.0057699, 0.0045384, 0.0057433, -0.0004489, 0.0005789
7: 0.9812666, 0.9820968, 0.9812349, 0.9820781, -0.0003141, 0.0004051
8: -0.0066494, -0.0057593, -0.0066833, -0.0057793, -0.0003368, 0.0004343
9: -0.0011953, -0.0006073, -0.0011821, -0.0005849, -0.0002869, 0.0002225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002111, upper bound: 0.0002084
time: 0.58 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002113, upper bound: 0.0002126
time: 0.56 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0017388, -0.0014253, -0.0017258, -0.0014084, -0.0001670, 0.0001188
1: -0.0087231, -0.0079274, -0.0086900, -0.0078847, -0.0004237, 0.0003014
2: 0.0296182, 0.0301118, 0.0296387, 0.0301383, -0.0002629, 0.0001870
3: 0.0033362, 0.0042579, 0.0032866, 0.0042196, -0.0003492, 0.0004909
4: -0.0077659, -0.0069566, -0.0077323, -0.0069131, -0.0004310, 0.0003066
5: 0.0107967, 0.0111032, 0.0108094, 0.0111197, -0.0001632, 0.0001161
6: 0.0046221, 0.0057919, 0.0045592, 0.0057433, -0.0004432, 0.0006230
7: 0.9812936, 0.9821122, 0.9812496, 0.9820782, -0.0003101, 0.0004359
8: -0.0066204, -0.0057428, -0.0066676, -0.0057793, -0.0003325, 0.0004674
9: -0.0012062, -0.0006264, -0.0011821, -0.0005953, -0.0003087, 0.0002196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002108, upper bound: 0.0002084
time: 0.58 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002109, upper bound: 0.0002126
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.74 seconds
IS_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001695
IS_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001695
IS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001764
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001764
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001656
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001656
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001710
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001964, upper bound: 0.0001710
IS_A1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001651
IS_A1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001681
IS_A1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001678
IS_A1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001740, upper bound: 0.0001727
IS_A1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001632, upper bound: 0.0001587
IS_A1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001632, upper bound: 0.0001627
IS_A1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001621
IS_A1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001685
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001831
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001866
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001826
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001866
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002080
IS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002125
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002147, upper bound: 0.0002080
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002148, upper bound: 0.0002125
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001831
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001866
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001691, upper bound: 0.0001826
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001866
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002111, upper bound: 0.0002084
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002113, upper bound: 0.0002126
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002108, upper bound: 0.0002084
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.74
Output dim: 7, lower bound: -0.0002109, upper bound: 0.0002126

## BFS IS instance: IS_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013930, -0.0017206, -0.0014136, -0.0001087, 0.0001371
1: -0.0086336, -0.0078454, -0.0086768, -0.0078979, -0.0002759, 0.0003480
2: 0.0296737, 0.0301627, 0.0296469, 0.0301301, -0.0001712, 0.0002159
3: 0.0032411, 0.0041542, 0.0033019, 0.0042042, -0.0004031, 0.0003196
4: -0.0076749, -0.0068731, -0.0077187, -0.0069265, -0.0002806, 0.0003539
5: 0.0108312, 0.0111348, 0.0108145, 0.0111146, -0.0001063, 0.0001341
6: 0.0045015, 0.0056603, 0.0045786, 0.0057238, -0.0005116, 0.0004056
7: 0.9812092, 0.9820201, 0.9812632, 0.9820644, -0.0003580, 0.0002838
8: -0.0067109, -0.0058415, -0.0066531, -0.0057939, -0.0003838, 0.0003043
9: -0.0011409, -0.0005667, -0.0011724, -0.0006049, -0.0002010, 0.0002535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001689
time: 0.54 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001695
time: 0.54 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013930, -0.0017283, -0.0014232, -0.0001052, 0.0001566
1: -0.0086336, -0.0078454, -0.0086963, -0.0079221, -0.0002669, 0.0003974
2: 0.0296737, 0.0301627, 0.0296348, 0.0301151, -0.0001656, 0.0002465
3: 0.0032411, 0.0041542, 0.0033299, 0.0042268, -0.0004604, 0.0003092
4: -0.0076749, -0.0068731, -0.0077386, -0.0069511, -0.0002715, 0.0004042
5: 0.0108312, 0.0111348, 0.0108070, 0.0111053, -0.0001028, 0.0001531
6: 0.0045015, 0.0056603, 0.0046142, 0.0057525, -0.0005843, 0.0003924
7: 0.9812092, 0.9820201, 0.9812881, 0.9820846, -0.0004088, 0.0002746
8: -0.0067109, -0.0058415, -0.0066264, -0.0057724, -0.0004383, 0.0002944
9: -0.0011409, -0.0005667, -0.0011866, -0.0006225, -0.0001945, 0.0002895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001689
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001695
time: 0.54 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013918, -0.0017233, -0.0014100, -0.0001112, 0.0001415
1: -0.0086336, -0.0078425, -0.0086836, -0.0078887, -0.0002821, 0.0003592
2: 0.0296737, 0.0301645, 0.0296427, 0.0301358, -0.0001750, 0.0002228
3: 0.0032377, 0.0041542, 0.0032913, 0.0042121, -0.0004161, 0.0003268
4: -0.0076749, -0.0068701, -0.0077257, -0.0069172, -0.0002869, 0.0003653
5: 0.0108311, 0.0111360, 0.0108119, 0.0111181, -0.0001087, 0.0001384
6: 0.0044972, 0.0056604, 0.0045652, 0.0057338, -0.0005281, 0.0004148
7: 0.9812063, 0.9820201, 0.9812537, 0.9820715, -0.0003695, 0.0002902
8: -0.0067142, -0.0058415, -0.0066632, -0.0057864, -0.0003962, 0.0003112
9: -0.0011410, -0.0005645, -0.0011774, -0.0005982, -0.0002055, 0.0002617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001750
time: 0.57 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001764
time: 0.55 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013918, -0.0017309, -0.0014199, -0.0001074, 0.0001623
1: -0.0086336, -0.0078425, -0.0087031, -0.0079138, -0.0002726, 0.0004117
2: 0.0296737, 0.0301645, 0.0296306, 0.0301203, -0.0001691, 0.0002555
3: 0.0032377, 0.0041542, 0.0033204, 0.0042347, -0.0004770, 0.0003157
4: -0.0076749, -0.0068701, -0.0077455, -0.0069427, -0.0002772, 0.0004188
5: 0.0108311, 0.0111360, 0.0108044, 0.0111085, -0.0001050, 0.0001586
6: 0.0044972, 0.0056604, 0.0046021, 0.0057625, -0.0006054, 0.0004007
7: 0.9812063, 0.9820201, 0.9812796, 0.9820916, -0.0004236, 0.0002804
8: -0.0067142, -0.0058415, -0.0066355, -0.0057649, -0.0004542, 0.0003006
9: -0.0011410, -0.0005645, -0.0011916, -0.0006165, -0.0001986, 0.0003000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001750
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001764
time: 0.55 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013979, -0.0017276, -0.0014234, -0.0001086, 0.0001487
1: -0.0086336, -0.0078581, -0.0086947, -0.0079227, -0.0002756, 0.0003774
2: 0.0296737, 0.0301549, 0.0296358, 0.0301147, -0.0001710, 0.0002342
3: 0.0032558, 0.0041542, 0.0033307, 0.0042249, -0.0004372, 0.0003193
4: -0.0076748, -0.0068860, -0.0077369, -0.0069517, -0.0002804, 0.0003839
5: 0.0108312, 0.0111300, 0.0108076, 0.0111050, -0.0001062, 0.0001454
6: 0.0045201, 0.0056603, 0.0046152, 0.0057501, -0.0005549, 0.0004053
7: 0.9812222, 0.9820201, 0.9812887, 0.9820829, -0.0003883, 0.0002836
8: -0.0066970, -0.0058415, -0.0066257, -0.0057742, -0.0004163, 0.0003040
9: -0.0011409, -0.0005759, -0.0011854, -0.0006230, -0.0002008, 0.0002750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001611
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001656
time: 0.54 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013979, -0.0017345, -0.0014332, -0.0001053, 0.0001673
1: -0.0086336, -0.0078581, -0.0087122, -0.0079475, -0.0002672, 0.0004247
2: 0.0296737, 0.0301549, 0.0296249, 0.0300994, -0.0001658, 0.0002635
3: 0.0032558, 0.0041542, 0.0033593, 0.0042453, -0.0004919, 0.0003096
4: -0.0076748, -0.0068860, -0.0077548, -0.0069769, -0.0002718, 0.0004319
5: 0.0108312, 0.0111300, 0.0108009, 0.0110955, -0.0001029, 0.0001636
6: 0.0045201, 0.0056603, 0.0046516, 0.0057759, -0.0006243, 0.0003929
7: 0.9812222, 0.9820201, 0.9813142, 0.9821010, -0.0004369, 0.0002749
8: -0.0066970, -0.0058415, -0.0065983, -0.0057548, -0.0004684, 0.0002947
9: -0.0011409, -0.0005759, -0.0011982, -0.0006410, -0.0001947, 0.0003094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001611
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001656
time: 0.54 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013969, -0.0017302, -0.0014204, -0.0001110, 0.0001502
1: -0.0086336, -0.0078553, -0.0087012, -0.0079150, -0.0002818, 0.0003812
2: 0.0296737, 0.0301565, 0.0296317, 0.0301195, -0.0001748, 0.0002365
3: 0.0032526, 0.0041542, 0.0033217, 0.0042325, -0.0004416, 0.0003264
4: -0.0076749, -0.0068832, -0.0077436, -0.0069439, -0.0002866, 0.0003877
5: 0.0108312, 0.0111310, 0.0108051, 0.0111080, -0.0001086, 0.0001469
6: 0.0045161, 0.0056603, 0.0046038, 0.0057597, -0.0005604, 0.0004142
7: 0.9812195, 0.9820201, 0.9812808, 0.9820896, -0.0003922, 0.0002899
8: -0.0067000, -0.0058415, -0.0066342, -0.0057669, -0.0004205, 0.0003108
9: -0.0011409, -0.0005739, -0.0011902, -0.0006174, -0.0002053, 0.0002777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001645
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001710
time: 0.53 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013969, -0.0017373, -0.0014302, -0.0001075, 0.0001716
1: -0.0086336, -0.0078553, -0.0087193, -0.0079400, -0.0002729, 0.0004354
2: 0.0296737, 0.0301565, 0.0296206, 0.0301040, -0.0001693, 0.0002701
3: 0.0032526, 0.0041542, 0.0033507, 0.0042534, -0.0005044, 0.0003161
4: -0.0076749, -0.0068832, -0.0077620, -0.0069694, -0.0002776, 0.0004429
5: 0.0108312, 0.0111310, 0.0107982, 0.0110984, -0.0001051, 0.0001677
6: 0.0045161, 0.0056603, 0.0046406, 0.0057863, -0.0006401, 0.0004012
7: 0.9812195, 0.9820201, 0.9813066, 0.9821082, -0.0004479, 0.0002807
8: -0.0067000, -0.0058415, -0.0066066, -0.0057470, -0.0004803, 0.0003010
9: -0.0011409, -0.0005739, -0.0012034, -0.0006356, -0.0001988, 0.0003172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001645
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001710
time: 0.55 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014084, -0.0017035, -0.0013924, -0.0001574, 0.0001131
1: -0.0086900, -0.0078846, -0.0086335, -0.0078441, -0.0003994, 0.0002871
2: 0.0296387, 0.0301384, 0.0296737, 0.0301635, -0.0002478, 0.0001781
3: 0.0032866, 0.0042195, 0.0032396, 0.0041541, -0.0003326, 0.0004627
4: -0.0077322, -0.0069130, -0.0076748, -0.0068718, -0.0004063, 0.0002921
5: 0.0108094, 0.0111197, 0.0108312, 0.0111353, -0.0001539, 0.0001106
6: 0.0045592, 0.0057432, 0.0044996, 0.0056603, -0.0004221, 0.0005873
7: 0.9812496, 0.9820781, 0.9812079, 0.9820200, -0.0002954, 0.0004109
8: -0.0066676, -0.0057793, -0.0067124, -0.0058416, -0.0003167, 0.0004406
9: -0.0011820, -0.0005952, -0.0011409, -0.0005657, -0.0002910, 0.0002092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001831
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001831
time: 0.54 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014071, -0.0017063, -0.0013892, -0.0001601, 0.0001178
1: -0.0086900, -0.0078813, -0.0086405, -0.0078358, -0.0004063, 0.0002990
2: 0.0296387, 0.0301405, 0.0296694, 0.0301686, -0.0002521, 0.0001855
3: 0.0032826, 0.0042195, 0.0032300, 0.0041621, -0.0003464, 0.0004707
4: -0.0077322, -0.0069096, -0.0076818, -0.0068634, -0.0004133, 0.0003041
5: 0.0108094, 0.0111210, 0.0108285, 0.0111385, -0.0001565, 0.0001152
6: 0.0045542, 0.0057433, 0.0044874, 0.0056704, -0.0004396, 0.0005974
7: 0.9812461, 0.9820781, 0.9811994, 0.9820272, -0.0003076, 0.0004180
8: -0.0066714, -0.0057793, -0.0067215, -0.0058340, -0.0003298, 0.0004482
9: -0.0011820, -0.0005928, -0.0011459, -0.0005597, -0.0002960, 0.0002178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001866
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001866
time: 0.55 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017317, -0.0014184, -0.0017035, -0.0013975, -0.0001663, 0.0001131
1: -0.0087050, -0.0079100, -0.0086335, -0.0078570, -0.0004220, 0.0002870
2: 0.0296294, 0.0301226, 0.0296737, 0.0301555, -0.0002618, 0.0001781
3: 0.0033159, 0.0042369, 0.0032545, 0.0041541, -0.0003325, 0.0004889
4: -0.0077475, -0.0069388, -0.0076748, -0.0068849, -0.0004293, 0.0002920
5: 0.0108036, 0.0111099, 0.0108312, 0.0111304, -0.0001626, 0.0001106
6: 0.0045965, 0.0057653, 0.0045185, 0.0056602, -0.0004220, 0.0006205
7: 0.9812757, 0.9820936, 0.9812211, 0.9820200, -0.0002953, 0.0004342
8: -0.0066397, -0.0057628, -0.0066982, -0.0058416, -0.0003166, 0.0004655
9: -0.0011930, -0.0006137, -0.0011409, -0.0005751, -0.0003075, 0.0002091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001826
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001826
time: 0.56 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017317, -0.0014173, -0.0017063, -0.0013943, -0.0001690, 0.0001180
1: -0.0087050, -0.0079072, -0.0086404, -0.0078489, -0.0004289, 0.0002995
2: 0.0296294, 0.0301244, 0.0296695, 0.0301605, -0.0002661, 0.0001858
3: 0.0033127, 0.0042369, 0.0032451, 0.0041621, -0.0003470, 0.0004969
4: -0.0077475, -0.0069360, -0.0076818, -0.0068767, -0.0004363, 0.0003047
5: 0.0108036, 0.0111110, 0.0108285, 0.0111335, -0.0001652, 0.0001154
6: 0.0045924, 0.0057653, 0.0045066, 0.0056704, -0.0004404, 0.0006306
7: 0.9812728, 0.9820936, 0.9812128, 0.9820271, -0.0003081, 0.0004413
8: -0.0066428, -0.0057627, -0.0067071, -0.0058340, -0.0003304, 0.0004731
9: -0.0011930, -0.0006117, -0.0011459, -0.0005692, -0.0003125, 0.0002182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001866
time: 0.54 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001866
time: 0.56 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014084, -0.0017258, -0.0014050, -0.0001207, 0.0001107
1: -0.0086900, -0.0078846, -0.0086899, -0.0078761, -0.0003062, 0.0002809
2: 0.0296387, 0.0301384, 0.0296387, 0.0301437, -0.0001900, 0.0001743
3: 0.0032866, 0.0042195, 0.0032766, 0.0042195, -0.0003255, 0.0003547
4: -0.0077322, -0.0069130, -0.0077322, -0.0069043, -0.0003114, 0.0002858
5: 0.0108094, 0.0111197, 0.0108094, 0.0111230, -0.0001180, 0.0001082
6: 0.0045592, 0.0057432, 0.0045466, 0.0057432, -0.0004130, 0.0004502
7: 0.9812496, 0.9820781, 0.9812408, 0.9820781, -0.0002890, 0.0003150
8: -0.0066676, -0.0057793, -0.0066771, -0.0057794, -0.0003099, 0.0003377
9: -0.0011820, -0.0005952, -0.0011820, -0.0005890, -0.0002231, 0.0002047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002080
time: 0.59 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002080
time: 0.59 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017258, -0.0014071, -0.0017288, -0.0014015, -0.0001237, 0.0001171
1: -0.0086900, -0.0078813, -0.0086976, -0.0078671, -0.0003140, 0.0002971
2: 0.0296387, 0.0301405, 0.0296340, 0.0301492, -0.0001948, 0.0001843
3: 0.0032826, 0.0042195, 0.0032663, 0.0042283, -0.0003441, 0.0003638
4: -0.0077322, -0.0069096, -0.0077399, -0.0068952, -0.0003194, 0.0003022
5: 0.0108094, 0.0111210, 0.0108065, 0.0111264, -0.0001210, 0.0001145
6: 0.0045542, 0.0057433, 0.0045335, 0.0057544, -0.0004368, 0.0004617
7: 0.9812461, 0.9820781, 0.9812316, 0.9820859, -0.0003056, 0.0003230
8: -0.0066714, -0.0057793, -0.0066869, -0.0057709, -0.0003277, 0.0003464
9: -0.0011820, -0.0005928, -0.0011876, -0.0005825, -0.0002288, 0.0002165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002142, upper bound: 0.0002125
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002142, upper bound: 0.0002125
time: 0.59 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017317, -0.0014184, -0.0017258, -0.0014105, -0.0001382, 0.0001092
1: -0.0087050, -0.0079100, -0.0086899, -0.0078898, -0.0003507, 0.0002771
2: 0.0296294, 0.0301226, 0.0296388, 0.0301351, -0.0002176, 0.0001719
3: 0.0033159, 0.0042369, 0.0032926, 0.0042194, -0.0003210, 0.0004063
4: -0.0077475, -0.0069388, -0.0077321, -0.0069183, -0.0003567, 0.0002819
5: 0.0108036, 0.0111099, 0.0108095, 0.0111177, -0.0001351, 0.0001068
6: 0.0045965, 0.0057653, 0.0045668, 0.0057431, -0.0004074, 0.0005156
7: 0.9812757, 0.9820936, 0.9812549, 0.9820780, -0.0002851, 0.0003608
8: -0.0066397, -0.0057628, -0.0066619, -0.0057794, -0.0003056, 0.0003868
9: -0.0011930, -0.0006137, -0.0011820, -0.0005990, -0.0002555, 0.0002019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002147, upper bound: 0.0002080
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002147, upper bound: 0.0002080
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017317, -0.0014173, -0.0017288, -0.0014074, -0.0001411, 0.0001158
1: -0.0087050, -0.0079072, -0.0086976, -0.0078821, -0.0003579, 0.0002939
2: 0.0296294, 0.0301244, 0.0296340, 0.0301399, -0.0002221, 0.0001823
3: 0.0033127, 0.0042369, 0.0032836, 0.0042283, -0.0003404, 0.0004147
4: -0.0077475, -0.0069360, -0.0077399, -0.0069104, -0.0003641, 0.0002989
5: 0.0108036, 0.0111110, 0.0108065, 0.0111207, -0.0001379, 0.0001132
6: 0.0045924, 0.0057653, 0.0045554, 0.0057544, -0.0004320, 0.0005263
7: 0.9812728, 0.9820936, 0.9812469, 0.9820859, -0.0003023, 0.0003683
8: -0.0066428, -0.0057627, -0.0066705, -0.0057710, -0.0003241, 0.0003948
9: -0.0011930, -0.0006117, -0.0011875, -0.0005934, -0.0002608, 0.0002141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002148, upper bound: 0.0002125
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002148, upper bound: 0.0002125
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014182, -0.0017035, -0.0013985, -0.0001710, 0.0001164
1: -0.0087081, -0.0079094, -0.0086335, -0.0078596, -0.0004340, 0.0002953
2: 0.0296275, 0.0301230, 0.0296737, 0.0301539, -0.0002693, 0.0001832
3: 0.0033152, 0.0042405, 0.0032575, 0.0041541, -0.0003421, 0.0005028
4: -0.0077506, -0.0069382, -0.0076748, -0.0068875, -0.0004415, 0.0003004
5: 0.0108025, 0.0111102, 0.0108312, 0.0111294, -0.0001672, 0.0001138
6: 0.0045956, 0.0057698, 0.0045223, 0.0056602, -0.0004342, 0.0006381
7: 0.9812750, 0.9820967, 0.9812238, 0.9820200, -0.0003038, 0.0004465
8: -0.0066403, -0.0057594, -0.0066953, -0.0058416, -0.0003257, 0.0004787
9: -0.0011952, -0.0006133, -0.0011409, -0.0005770, -0.0003162, 0.0002152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001831
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001831
time: 0.55 seconds

## BFS IS instance: IS_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014169, -0.0017063, -0.0013951, -0.0001734, 0.0001209
1: -0.0087081, -0.0079062, -0.0086404, -0.0078510, -0.0004401, 0.0003068
2: 0.0296275, 0.0301250, 0.0296695, 0.0301593, -0.0002731, 0.0001903
3: 0.0033116, 0.0042405, 0.0032475, 0.0041621, -0.0003554, 0.0005099
4: -0.0077506, -0.0069350, -0.0076818, -0.0068788, -0.0004477, 0.0003121
5: 0.0108024, 0.0111114, 0.0108285, 0.0111327, -0.0001696, 0.0001182
6: 0.0045909, 0.0057699, 0.0045097, 0.0056704, -0.0004511, 0.0006471
7: 0.9812719, 0.9820968, 0.9812149, 0.9820271, -0.0003156, 0.0004528
8: -0.0066438, -0.0057593, -0.0067048, -0.0058340, -0.0003384, 0.0004855
9: -0.0011952, -0.0006110, -0.0011459, -0.0005707, -0.0003207, 0.0002235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001866
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001866
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017388, -0.0014284, -0.0017035, -0.0014037, -0.0001789, 0.0001166
1: -0.0087231, -0.0079355, -0.0086335, -0.0078726, -0.0004541, 0.0002958
2: 0.0296182, 0.0301068, 0.0296738, 0.0301458, -0.0002817, 0.0001835
3: 0.0033454, 0.0042578, 0.0032727, 0.0041541, -0.0003426, 0.0005260
4: -0.0077658, -0.0069647, -0.0076748, -0.0069008, -0.0004619, 0.0003008
5: 0.0107967, 0.0111001, 0.0108312, 0.0111243, -0.0001749, 0.0001140
6: 0.0046339, 0.0057918, 0.0045416, 0.0056602, -0.0004349, 0.0006676
7: 0.9813018, 0.9821122, 0.9812372, 0.9820200, -0.0003043, 0.0004672
8: -0.0066116, -0.0057429, -0.0066809, -0.0058416, -0.0003262, 0.0005009
9: -0.0012061, -0.0006323, -0.0011409, -0.0005865, -0.0003309, 0.0002155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001691, upper bound: 0.0001826
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001691, upper bound: 0.0001826
time: 0.57 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017388, -0.0014273, -0.0017062, -0.0014004, -0.0001812, 0.0001212
1: -0.0087231, -0.0079326, -0.0086404, -0.0078643, -0.0004597, 0.0003076
2: 0.0296182, 0.0301086, 0.0296695, 0.0301510, -0.0002852, 0.0001909
3: 0.0033422, 0.0042578, 0.0032630, 0.0041621, -0.0003564, 0.0005326
4: -0.0077659, -0.0069619, -0.0076818, -0.0068923, -0.0004676, 0.0003129
5: 0.0107967, 0.0111012, 0.0108285, 0.0111276, -0.0001771, 0.0001185
6: 0.0046298, 0.0057919, 0.0045292, 0.0056704, -0.0004523, 0.0006759
7: 0.9812990, 0.9821122, 0.9812286, 0.9820271, -0.0003165, 0.0004730
8: -0.0066147, -0.0057428, -0.0066901, -0.0058340, -0.0003393, 0.0005071
9: -0.0012061, -0.0006302, -0.0011459, -0.0005804, -0.0003350, 0.0002242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001866
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001866
time: 0.55 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014182, -0.0017258, -0.0014110, -0.0001454, 0.0001150
1: -0.0087081, -0.0079094, -0.0086899, -0.0078913, -0.0003689, 0.0002919
2: 0.0296275, 0.0301230, 0.0296388, 0.0301342, -0.0002289, 0.0001811
3: 0.0033152, 0.0042405, 0.0032942, 0.0042194, -0.0003381, 0.0004273
4: -0.0077506, -0.0069382, -0.0077321, -0.0069198, -0.0003752, 0.0002969
5: 0.0108025, 0.0111102, 0.0108095, 0.0111172, -0.0001421, 0.0001124
6: 0.0045956, 0.0057698, 0.0045689, 0.0057431, -0.0004291, 0.0005423
7: 0.9812750, 0.9820967, 0.9812564, 0.9820780, -0.0003003, 0.0003795
8: -0.0066403, -0.0057594, -0.0066603, -0.0057794, -0.0003219, 0.0004069
9: -0.0011952, -0.0006133, -0.0011820, -0.0006001, -0.0002688, 0.0002127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002111, upper bound: 0.0002081
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002111, upper bound: 0.0002084
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017329, -0.0014169, -0.0017288, -0.0014078, -0.0001477, 0.0001204
1: -0.0087081, -0.0079062, -0.0086976, -0.0078830, -0.0003749, 0.0003055
2: 0.0296275, 0.0301250, 0.0296340, 0.0301394, -0.0002326, 0.0001895
3: 0.0033116, 0.0042405, 0.0032846, 0.0042283, -0.0003538, 0.0004343
4: -0.0077506, -0.0069350, -0.0077399, -0.0069113, -0.0003813, 0.0003107
5: 0.0108024, 0.0111114, 0.0108065, 0.0111204, -0.0001444, 0.0001177
6: 0.0045909, 0.0057699, 0.0045567, 0.0057544, -0.0004491, 0.0005512
7: 0.9812719, 0.9820968, 0.9812478, 0.9820859, -0.0003142, 0.0003857
8: -0.0066438, -0.0057593, -0.0066695, -0.0057710, -0.0003369, 0.0004135
9: -0.0011952, -0.0006110, -0.0011875, -0.0005940, -0.0002731, 0.0002226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002112, upper bound: 0.0002125
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002112, upper bound: 0.0002126
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017388, -0.0014284, -0.0017258, -0.0014165, -0.0001571, 0.0001134
1: -0.0087231, -0.0079355, -0.0086899, -0.0079051, -0.0003988, 0.0002877
2: 0.0296182, 0.0301068, 0.0296388, 0.0301257, -0.0002474, 0.0001785
3: 0.0033454, 0.0042578, 0.0033102, 0.0042194, -0.0003333, 0.0004620
4: -0.0077658, -0.0069647, -0.0077321, -0.0069338, -0.0004056, 0.0002927
5: 0.0107967, 0.0111001, 0.0108095, 0.0111118, -0.0001536, 0.0001109
6: 0.0046339, 0.0057918, 0.0045892, 0.0057431, -0.0004230, 0.0005863
7: 0.9813018, 0.9821122, 0.9812706, 0.9820781, -0.0002960, 0.0004103
8: -0.0066116, -0.0057429, -0.0066451, -0.0057794, -0.0003174, 0.0004399
9: -0.0012061, -0.0006323, -0.0011820, -0.0006101, -0.0002906, 0.0002096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002107, upper bound: 0.0002084
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002107, upper bound: 0.0002084
time: 0.63 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017388, -0.0014273, -0.0017288, -0.0014134, -0.0001594, 0.0001190
1: -0.0087231, -0.0079326, -0.0086976, -0.0078973, -0.0004046, 0.0003020
2: 0.0296182, 0.0301086, 0.0296340, 0.0301305, -0.0002510, 0.0001873
3: 0.0033422, 0.0042578, 0.0033012, 0.0042283, -0.0003498, 0.0004687
4: -0.0077659, -0.0069619, -0.0077399, -0.0069259, -0.0004116, 0.0003072
5: 0.0107967, 0.0111012, 0.0108065, 0.0111148, -0.0001559, 0.0001163
6: 0.0046298, 0.0057919, 0.0045777, 0.0057544, -0.0004440, 0.0005949
7: 0.9812990, 0.9821122, 0.9812626, 0.9820859, -0.0003107, 0.0004163
8: -0.0066147, -0.0057428, -0.0066537, -0.0057710, -0.0003331, 0.0004463
9: -0.0012061, -0.0006302, -0.0011875, -0.0006044, -0.0002948, 0.0002200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002109, upper bound: 0.0002126
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002109, upper bound: 0.0002126
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.70 seconds
IS_A1_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001689
IS_A1_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001695
IS_A1_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001689
IS_A1_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001695
IS_A1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001750
IS_A1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001764
IS_A1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001750
IS_A1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001799, upper bound: 0.0001764
IS_A1_A1_B2_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001611
IS_A1_A1_B2_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001656
IS_A1_A1_B2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001611
IS_A1_A1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001749, upper bound: 0.0001656
IS_A1_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001645
IS_A1_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001710
IS_A1_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001645
IS_A1_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001710
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001831
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001831
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001866
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001847, upper bound: 0.0001866
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001826
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001826
IS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001866
IS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001866
IS_A2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002080
IS_A2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002080
IS_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002142, upper bound: 0.0002125
IS_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002142, upper bound: 0.0002125
IS_A2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002147, upper bound: 0.0002080
IS_A2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002147, upper bound: 0.0002080
IS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002148, upper bound: 0.0002125
IS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002148, upper bound: 0.0002125
IS_A2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001831
IS_A2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001831
IS_A2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001866
IS_A2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001866
IS_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001691, upper bound: 0.0001826
IS_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001691, upper bound: 0.0001826
IS_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001866
IS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001866
IS_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002111, upper bound: 0.0002081
IS_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002111, upper bound: 0.0002084
IS_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002112, upper bound: 0.0002125
IS_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002112, upper bound: 0.0002126
IS_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002107, upper bound: 0.0002084
IS_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002107, upper bound: 0.0002084
IS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002109, upper bound: 0.0002126
IS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.70
Output dim: 7, lower bound: -0.0002109, upper bound: 0.0002126

## BFS IS instance: IS_A1_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013930, -0.0017035, -0.0014023, -0.0001033, 0.0001128
1: -0.0086336, -0.0078454, -0.0086335, -0.0078692, -0.0002622, 0.0002863
2: 0.0296737, 0.0301627, 0.0296738, 0.0301479, -0.0001626, 0.0001776
3: 0.0032411, 0.0041542, 0.0032687, 0.0041541, -0.0003317, 0.0003037
4: -0.0076749, -0.0068731, -0.0076747, -0.0068973, -0.0002667, 0.0002913
5: 0.0108312, 0.0111348, 0.0108312, 0.0111257, -0.0001010, 0.0001103
6: 0.0045015, 0.0056603, 0.0045365, 0.0056602, -0.0004210, 0.0003854
7: 0.9812092, 0.9820201, 0.9812337, 0.9820200, -0.0002946, 0.0002697
8: -0.0067109, -0.0058415, -0.0066847, -0.0058416, -0.0003158, 0.0002892
9: -0.0011409, -0.0005667, -0.0011409, -0.0005840, -0.0001910, 0.0002086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 194
type: B, layer: 3, pos: 194
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 83
type: B, layer: 3, pos: 83
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 194

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001693, upper bound: 0.0001640
time: 0.54 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001638
time: 0.59 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013930, -0.0017257, -0.0014135, -0.0001085, 0.0001528
1: -0.0086336, -0.0078454, -0.0086899, -0.0078976, -0.0002754, 0.0003878
2: 0.0296737, 0.0301627, 0.0296388, 0.0301303, -0.0001708, 0.0002406
3: 0.0032411, 0.0041542, 0.0033016, 0.0042194, -0.0004492, 0.0003190
4: -0.0076749, -0.0068731, -0.0077321, -0.0069262, -0.0002801, 0.0003945
5: 0.0108312, 0.0111348, 0.0108095, 0.0111147, -0.0001061, 0.0001494
6: 0.0045015, 0.0056603, 0.0045783, 0.0057431, -0.0005701, 0.0004048
7: 0.9812092, 0.9820201, 0.9812629, 0.9820780, -0.0003990, 0.0002833
8: -0.0067109, -0.0058415, -0.0066533, -0.0057794, -0.0004277, 0.0003037
9: -0.0011409, -0.0005667, -0.0011820, -0.0006047, -0.0002006, 0.0002826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 194
type: A, layer: 3, pos: 194
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 83
type: B, layer: 3, pos: 83
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 194

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001624, upper bound: 0.0001703
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001642
time: 0.58 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0017036, -0.0013930, -0.0017113, -0.0014105, -0.0001097, 0.0001384
1: -0.0086336, -0.0078454, -0.0086534, -0.0078900, -0.0002785, 0.0003511
2: 0.0296737, 0.0301627, 0.0296614, 0.0301350, -0.0001728, 0.0002178
3: 0.0032411, 0.0041542, 0.0032928, 0.0041771, -0.0004067, 0.0003226
4: -0.0076749, -0.0068731, -0.0076949, -0.0069185, -0.0002833, 0.0003571
5: 0.0108312, 0.0111348, 0.0108235, 0.0111176, -0.0001073, 0.0001353
6: 0.0045015, 0.0056603, 0.0045671, 0.0056894, -0.0005162, 0.0004094
7: 0.9812092, 0.9820201, 0.9812551, 0.9820405, -0.0003612, 0.0002865
8: -0.0067109, -0.0058415, -0.0066617, -0.0058197, -0.0003873, 0.0003072
9: -0.0011409, -0.0005667, -0.0011553, -0.0005992, -0.0002029, 0.0002558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 194
type: A, layer: 3, pos: 194
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 83
type: A, layer: 3, pos: 83
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 194

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 194

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 222

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 222

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 83

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 83

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5

No IS candidates found

### IS candidates at layer 7

No IS candidates found

No IS candidates found

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.83 + 164.85 = 167.68 seconds
