## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00221263


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0057605, 0.0101726, 0.0057605, 0.0101726, -0.0044121, 0.0044121)
1: (0.0011233, 0.0026187, 0.0011233, 0.0026187, -0.0014953, 0.0014953)
2: (0.0084098, 0.0111750, 0.0084098, 0.0111750, -0.0027652, 0.0027652)
3: (-0.0049598, -0.0021757, -0.0049598, -0.0021757, -0.0026555, 0.0026555)
4: (-0.0007440, 0.0013324, -0.0007440, 0.0013324, -0.0018098, 0.0018098)
5: (0.0024132, 0.0047345, 0.0024132, 0.0047345, -0.0023111, 0.0023111)
6: (-0.0113413, -0.0035150, -0.0113413, -0.0035150, -0.0069635, 0.0069635)
7: (0.0012241, 0.0124005, 0.0012241, 0.0124005, -0.0099231, 0.0099231)
8: (0.9907851, 0.9980579, 0.9907851, 0.9980579, -0.0062839, 0.0062839)
9: (-0.0140256, -0.0071946, -0.0140256, -0.0071946, -0.0059670, 0.0059670)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.92 + 1.60 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0029343, upper bound: 0.0029343

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026744, upper bound: 0.0027676
time: 0.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027729, upper bound: 0.0027729
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 8, lower bound: -0.0026744, upper bound: 0.0027676
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 8, lower bound: -0.0027729, upper bound: 0.0027729

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0061037, 0.0100640, 0.0059110, 0.0101281, -0.0040244, 0.0041530
1: 0.0010929, 0.0026039, 0.0011447, 0.0026183, -0.0015210, 0.0014592
2: 0.0084422, 0.0109853, 0.0084408, 0.0110918, -0.0026497, 0.0025445
3: -0.0049015, -0.0022921, -0.0049583, -0.0022583, -0.0024874, 0.0025132
4: -0.0005383, 0.0012692, -0.0006508, 0.0013307, -0.0016045, 0.0016366
5: 0.0024903, 0.0045335, 0.0024334, 0.0046464, -0.0021561, 0.0020957
6: -0.0110761, -0.0043128, -0.0113127, -0.0038647, -0.0063464, 0.0061612
7: 0.0022313, 0.0120776, 0.0016958, 0.0123920, -0.0088878, 0.0090248
8: 0.9915504, 0.9978076, 0.9911205, 0.9980447, -0.0055261, 0.0056875
9: -0.0138191, -0.0078639, -0.0140202, -0.0074990, -0.0053992, 0.0052970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026744, upper bound: 0.0026744
time: 0.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026744, upper bound: 0.0027676
time: 0.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0059307, 0.0101180, 0.0058327, 0.0101494, -0.0041357, 0.0042854
1: 0.0011431, 0.0026180, 0.0011321, 0.0026184, -0.0014532, 0.0014858
2: 0.0084441, 0.0110809, 0.0084246, 0.0111351, -0.0026910, 0.0026563
3: -0.0049570, -0.0022768, -0.0049587, -0.0022176, -0.0025952, 0.0024350
4: -0.0006371, 0.0013292, -0.0006986, 0.0013311, -0.0015785, 0.0017611
5: 0.0024416, 0.0046348, 0.0024252, 0.0046923, -0.0022506, 0.0020750
6: -0.0112920, -0.0039107, -0.0113208, -0.0036827, -0.0067807, 0.0060938
7: 0.0017715, 0.0123846, 0.0014589, 0.0123941, -0.0087245, 0.0096615
8: 0.9911646, 0.9980339, 0.9909459, 0.9980482, -0.0054614, 0.0061170
9: -0.0140154, -0.0075443, -0.0140215, -0.0073435, -0.0058071, 0.0052100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027676, upper bound: 0.0026744
time: 0.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027676, upper bound: 0.0027729
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 8, lower bound: -0.0026744, upper bound: 0.0026744
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 8, lower bound: -0.0026744, upper bound: 0.0027676
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 8, lower bound: -0.0027676, upper bound: 0.0026744
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 8, lower bound: -0.0027676, upper bound: 0.0027729

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061037, 0.0100640, 0.0061037, 0.0100640, -0.0039603, 0.0039603
1: 0.0010929, 0.0026039, 0.0010929, 0.0026039, -0.0015042, 0.0015042
2: 0.0084422, 0.0109853, 0.0084422, 0.0109853, -0.0025431, 0.0025431
3: -0.0049015, -0.0022921, -0.0049015, -0.0022921, -0.0024467, 0.0024467
4: -0.0005383, 0.0012692, -0.0005383, 0.0012692, -0.0015325, 0.0015325
5: 0.0024903, 0.0045335, 0.0024903, 0.0045335, -0.0020431, 0.0020431
6: -0.0110761, -0.0043128, -0.0110761, -0.0043128, -0.0059321, 0.0059321
7: 0.0022313, 0.0120776, 0.0022313, 0.0120776, -0.0085196, 0.0085196
8: 0.9915504, 0.9978076, 0.9915504, 0.9978076, -0.0052901, 0.0052901
9: -0.0138191, -0.0078639, -0.0138191, -0.0078639, -0.0050616, 0.0050616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025041, upper bound: 0.0025652
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025700, upper bound: 0.0025703
time: 0.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061037, 0.0100640, 0.0059307, 0.0101180, -0.0040143, 0.0041333
1: 0.0010929, 0.0026039, 0.0011431, 0.0026180, -0.0015193, 0.0014573
2: 0.0084422, 0.0109853, 0.0084441, 0.0110809, -0.0026387, 0.0025411
3: -0.0049015, -0.0022921, -0.0049570, -0.0022768, -0.0024669, 0.0025063
4: -0.0005383, 0.0012692, -0.0006371, 0.0013292, -0.0015970, 0.0016440
5: 0.0024903, 0.0045335, 0.0024416, 0.0046348, -0.0021445, 0.0020918
6: -0.0110761, -0.0043128, -0.0112920, -0.0039107, -0.0063736, 0.0061480
7: 0.0022313, 0.0120776, 0.0017715, 0.0123846, -0.0088496, 0.0090510
8: 0.9915504, 0.9978076, 0.9911646, 0.9980339, -0.0055100, 0.0057136
9: -0.0138191, -0.0078639, -0.0140154, -0.0075443, -0.0054232, 0.0052726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025041, upper bound: 0.0026572
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025700, upper bound: 0.0026619
time: 0.77 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0059307, 0.0101180, 0.0061037, 0.0100640, -0.0041333, 0.0040143
1: 0.0011431, 0.0026180, 0.0010929, 0.0026039, -0.0014573, 0.0015193
2: 0.0084441, 0.0110809, 0.0084422, 0.0109853, -0.0025411, 0.0026387
3: -0.0049570, -0.0022768, -0.0049015, -0.0022921, -0.0025063, 0.0024669
4: -0.0006371, 0.0013292, -0.0005383, 0.0012692, -0.0016440, 0.0015970
5: 0.0024416, 0.0046348, 0.0024903, 0.0045335, -0.0020918, 0.0021445
6: -0.0112920, -0.0039107, -0.0110761, -0.0043128, -0.0061480, 0.0063736
7: 0.0017715, 0.0123846, 0.0022313, 0.0120776, -0.0090510, 0.0088496
8: 0.9911646, 0.9980339, 0.9915504, 0.9978076, -0.0057136, 0.0055100
9: -0.0140154, -0.0075443, -0.0138191, -0.0078639, -0.0052726, 0.0054232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025938, upper bound: 0.0025647
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026619, upper bound: 0.0025700
time: 0.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0059307, 0.0101180, 0.0059307, 0.0101180, -0.0041096, 0.0041096
1: 0.0011431, 0.0026180, 0.0011431, 0.0026180, -0.0014522, 0.0014522
2: 0.0084441, 0.0110809, 0.0084441, 0.0110809, -0.0026368, 0.0026368
3: -0.0049570, -0.0022768, -0.0049570, -0.0022768, -0.0024310, 0.0024310
4: -0.0006371, 0.0013292, -0.0006371, 0.0013292, -0.0015741, 0.0015741
5: 0.0024416, 0.0046348, 0.0024416, 0.0046348, -0.0020619, 0.0020619
6: -0.0112920, -0.0039107, -0.0112920, -0.0039107, -0.0060671, 0.0060670
7: 0.0017715, 0.0123846, 0.0017715, 0.0123846, -0.0087019, 0.0087019
8: 0.9911646, 0.9980339, 0.9911646, 0.9980339, -0.0054454, 0.0054454
9: -0.0140154, -0.0075443, -0.0140154, -0.0075443, -0.0051955, 0.0051955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025938, upper bound: 0.0025754
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026619, upper bound: 0.0025808
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.49 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0025041, upper bound: 0.0025652
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0025700, upper bound: 0.0025703
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0025041, upper bound: 0.0026572
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0025700, upper bound: 0.0026619
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0025938, upper bound: 0.0025647
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0026619, upper bound: 0.0025700
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0025938, upper bound: 0.0025754
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0026619, upper bound: 0.0025808

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062809, 0.0092052, 0.0061781, 0.0096586, -0.0033776, 0.0027970
1: 0.0017948, 0.0026036, 0.0013845, 0.0026038, -0.0007769, 0.0012071
2: 0.0091855, 0.0108873, 0.0087930, 0.0109442, -0.0016717, 0.0020943
3: -0.0049000, -0.0030148, -0.0049008, -0.0025892, -0.0021199, 0.0016476
4: -0.0003725, 0.0012676, -0.0004691, 0.0012685, -0.0013471, 0.0014576
5: 0.0027780, 0.0044297, 0.0026238, 0.0044900, -0.0015272, 0.0017310
6: -0.0108566, -0.0047247, -0.0109849, -0.0044855, -0.0053609, 0.0053062
7: 0.0034355, 0.0120692, 0.0027278, 0.0120737, -0.0071637, 0.0079680
8: 0.9919455, 0.9977508, 0.9917161, 0.9977839, -0.0048046, 0.0050104
9: -0.0138137, -0.0084328, -0.0138166, -0.0081010, -0.0048044, 0.0044252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0023214
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024314, upper bound: 0.0025060
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062269, 0.0096573, 0.0061629, 0.0098194, -0.0035925, 0.0034226
1: 0.0014370, 0.0026186, 0.0012634, 0.0026038, -0.0011426, 0.0013529
2: 0.0088207, 0.0109172, 0.0086532, 0.0109526, -0.0021318, 0.0022640
3: -0.0049593, -0.0026515, -0.0049008, -0.0024736, -0.0023339, 0.0020103
4: -0.0004372, 0.0013318, -0.0004885, 0.0012684, -0.0013949, 0.0015711
5: 0.0026062, 0.0044614, 0.0025705, 0.0044988, -0.0017942, 0.0018262
6: -0.0111736, -0.0045990, -0.0110213, -0.0044502, -0.0058700, 0.0054175
7: 0.0028995, 0.0123977, 0.0025656, 0.0120737, -0.0076155, 0.0086182
8: 0.9918250, 0.9980057, 0.9916822, 0.9977924, -0.0048769, 0.0054103
9: -0.0140238, -0.0082056, -0.0138166, -0.0080328, -0.0051807, 0.0045972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024224, upper bound: 0.0023283
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025124, upper bound: 0.0025124
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062809, 0.0092052, 0.0060038, 0.0097351, -0.0034186, 0.0029881
1: 0.0017948, 0.0026036, 0.0014216, 0.0026178, -0.0007920, 0.0011725
2: 0.0091855, 0.0108873, 0.0087661, 0.0110405, -0.0017774, 0.0021212
3: -0.0049000, -0.0030148, -0.0049562, -0.0025583, -0.0021576, 0.0017072
4: -0.0003725, 0.0012676, -0.0005709, 0.0013285, -0.0014116, 0.0015693
5: 0.0027780, 0.0044297, 0.0025743, 0.0045920, -0.0016391, 0.0017727
6: -0.0108566, -0.0047247, -0.0112081, -0.0040804, -0.0058050, 0.0055154
7: 0.0034355, 0.0120692, 0.0022467, 0.0123806, -0.0074936, 0.0085092
8: 0.9919455, 0.9977508, 0.9913275, 0.9980075, -0.0050295, 0.0054365
9: -0.0138137, -0.0084328, -0.0140128, -0.0077708, -0.0051667, 0.0046362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0023910
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024314, upper bound: 0.0026035
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062269, 0.0096573, 0.0059953, 0.0098643, -0.0036185, 0.0036129
1: 0.0014370, 0.0026186, 0.0013188, 0.0026178, -0.0011577, 0.0012997
2: 0.0088207, 0.0109172, 0.0086549, 0.0110452, -0.0022245, 0.0022622
3: -0.0049593, -0.0026515, -0.0049562, -0.0024585, -0.0023559, 0.0020698
4: -0.0004372, 0.0013318, -0.0005849, 0.0013284, -0.0014594, 0.0016827
5: 0.0026062, 0.0044614, 0.0025277, 0.0045970, -0.0019056, 0.0018504
6: -0.0111736, -0.0045990, -0.0112364, -0.0040608, -0.0063123, 0.0056202
7: 0.0028995, 0.0123977, 0.0021182, 0.0123803, -0.0079452, 0.0091482
8: 0.9918250, 0.9980057, 0.9913085, 0.9980157, -0.0050972, 0.0058347
9: -0.0140238, -0.0082056, -0.0140127, -0.0077208, -0.0055427, 0.0048081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024221, upper bound: 0.0023980
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025120, upper bound: 0.0026087
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061063, 0.0092904, 0.0061781, 0.0096586, -0.0035523, 0.0028679
1: 0.0018120, 0.0026175, 0.0013845, 0.0026038, -0.0007585, 0.0012213
2: 0.0091464, 0.0109839, 0.0087930, 0.0109442, -0.0017003, 0.0021908
3: -0.0049551, -0.0029583, -0.0049008, -0.0025892, -0.0021758, 0.0017153
4: -0.0004765, 0.0013272, -0.0004691, 0.0012685, -0.0014607, 0.0015187
5: 0.0027283, 0.0045320, 0.0026238, 0.0044900, -0.0015766, 0.0018425
6: -0.0110875, -0.0043187, -0.0109849, -0.0044855, -0.0055776, 0.0057485
7: 0.0029324, 0.0123743, 0.0027278, 0.0120737, -0.0077207, 0.0082797
8: 0.9915560, 0.9979685, 0.9917161, 0.9977839, -0.0052290, 0.0052294
9: -0.0140088, -0.0080948, -0.0138166, -0.0081010, -0.0050042, 0.0047949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024393, upper bound: 0.0023214
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025332, upper bound: 0.0025053
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0060627, 0.0097096, 0.0061629, 0.0098194, -0.0037567, 0.0034544
1: 0.0014860, 0.0026308, 0.0012634, 0.0026038, -0.0010966, 0.0013671
2: 0.0088089, 0.0110079, 0.0086532, 0.0109526, -0.0021308, 0.0023548
3: -0.0050076, -0.0026308, -0.0049008, -0.0024736, -0.0023903, 0.0020436
4: -0.0005315, 0.0013841, -0.0004885, 0.0012684, -0.0015053, 0.0016327
5: 0.0025663, 0.0045575, 0.0025705, 0.0044988, -0.0018314, 0.0019373
6: -0.0113670, -0.0042175, -0.0110213, -0.0044502, -0.0060621, 0.0058584
7: 0.0024679, 0.0126650, 0.0025656, 0.0120737, -0.0081398, 0.0089321
8: 0.9914590, 0.9981931, 0.9916822, 0.9977924, -0.0052999, 0.0056268
9: -0.0141947, -0.0079014, -0.0138166, -0.0080328, -0.0053822, 0.0049546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025210, upper bound: 0.0023283
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026087, upper bound: 0.0025120
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061063, 0.0092904, 0.0060038, 0.0097351, -0.0033693, 0.0028105
1: 0.0018120, 0.0026175, 0.0014216, 0.0026178, -0.0007534, 0.0011668
2: 0.0091464, 0.0109839, 0.0087661, 0.0110405, -0.0016669, 0.0021383
3: -0.0049551, -0.0029583, -0.0049562, -0.0025583, -0.0021141, 0.0016600
4: -0.0004765, 0.0013272, -0.0005709, 0.0013285, -0.0013906, 0.0014969
5: 0.0027283, 0.0045320, 0.0025743, 0.0045920, -0.0015473, 0.0017362
6: -0.0110875, -0.0043187, -0.0112081, -0.0040804, -0.0055119, 0.0054495
7: 0.0029324, 0.0123743, 0.0022467, 0.0123806, -0.0073684, 0.0081443
8: 0.9915560, 0.9979685, 0.9913275, 0.9980075, -0.0049626, 0.0051625
9: -0.0140088, -0.0080948, -0.0140128, -0.0077708, -0.0049310, 0.0045661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024396, upper bound: 0.0023213
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025378, upper bound: 0.0025164
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0060627, 0.0097096, 0.0059953, 0.0098643, -0.0035790, 0.0033853
1: 0.0014860, 0.0026308, 0.0013188, 0.0026178, -0.0010901, 0.0012931
2: 0.0088089, 0.0110079, 0.0086549, 0.0110452, -0.0021061, 0.0022996
3: -0.0050076, -0.0026308, -0.0049562, -0.0024585, -0.0023083, 0.0019957
4: -0.0005315, 0.0013841, -0.0005849, 0.0013284, -0.0014361, 0.0016079
5: 0.0025663, 0.0045575, 0.0025277, 0.0045970, -0.0017947, 0.0018165
6: -0.0113670, -0.0042175, -0.0112364, -0.0040608, -0.0060107, 0.0055517
7: 0.0024679, 0.0126650, 0.0021182, 0.0123803, -0.0077938, 0.0087713
8: 0.9914590, 0.9981931, 0.9913085, 0.9980157, -0.0050329, 0.0055618
9: -0.0141947, -0.0079014, -0.0140127, -0.0077208, -0.0052986, 0.0047293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025221, upper bound: 0.0023304
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026132, upper bound: 0.0025212
time: 0.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0023214
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0024314, upper bound: 0.0025060
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0024224, upper bound: 0.0023283
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0025124, upper bound: 0.0025124
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0023910
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0024314, upper bound: 0.0026035
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0024221, upper bound: 0.0023980
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0025120, upper bound: 0.0026087
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0024393, upper bound: 0.0023214
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0025332, upper bound: 0.0025053
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0025210, upper bound: 0.0023283
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0026087, upper bound: 0.0025120
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0024396, upper bound: 0.0023213
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0025378, upper bound: 0.0025164
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0025221, upper bound: 0.0023304
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 8, lower bound: -0.0026132, upper bound: 0.0025212

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063776, 0.0088980, 0.0063827, 0.0088602, -0.0019555, 0.0020438
1: 0.0021940, 0.0026035, 0.0022444, 0.0026023, -0.0003369, 0.0002889
2: 0.0094350, 0.0108339, 0.0094613, 0.0108310, -0.0011398, 0.0010811
3: -0.0048997, -0.0034268, -0.0048952, -0.0034785, -0.0011434, 0.0011719
4: -0.0002793, 0.0012673, -0.0002713, 0.0012623, -0.0012158, 0.0012379
5: 0.0029015, 0.0043731, 0.0029187, 0.0043701, -0.0011885, 0.0011455
6: -0.0107519, -0.0049493, -0.0107196, -0.0049612, -0.0046663, 0.0045450
7: 0.0041275, 0.0120676, 0.0042001, 0.0120424, -0.0062516, 0.0063300
8: 0.9921610, 0.9977192, 0.9921725, 0.9976968, -0.0043603, 0.0044651
9: -0.0138127, -0.0087533, -0.0137966, -0.0087820, -0.0040476, 0.0039782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022549
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022773, upper bound: 0.0022684
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063075, 0.0091240, 0.0062416, 0.0093425, -0.0028370, 0.0024515
1: 0.0018924, 0.0026035, 0.0016423, 0.0026036, -0.0006731, 0.0009240
2: 0.0092619, 0.0108726, 0.0090745, 0.0109090, -0.0014513, 0.0017567
3: -0.0048998, -0.0031176, -0.0049003, -0.0028513, -0.0017607, 0.0015391
4: -0.0003479, 0.0012673, -0.0004110, 0.0012679, -0.0013255, 0.0013154
5: 0.0028096, 0.0044141, 0.0027269, 0.0044527, -0.0013546, 0.0015104
6: -0.0108276, -0.0047863, -0.0109032, -0.0046333, -0.0048707, 0.0050890
7: 0.0036116, 0.0120680, 0.0031529, 0.0120707, -0.0069947, 0.0070924
8: 0.9920046, 0.9977425, 0.9918578, 0.9977633, -0.0047197, 0.0045783
9: -0.0138130, -0.0085167, -0.0138147, -0.0083002, -0.0043283, 0.0043502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0024261
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0024458
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063324, 0.0092319, 0.0063772, 0.0089098, -0.0020571, 0.0025851
1: 0.0018506, 0.0026185, 0.0021676, 0.0026024, -0.0006976, 0.0003972
2: 0.0091946, 0.0108588, 0.0094244, 0.0108341, -0.0015245, 0.0011515
3: -0.0049590, -0.0030887, -0.0048952, -0.0034033, -0.0013211, 0.0015148
4: -0.0003378, 0.0013314, -0.0002819, 0.0012624, -0.0012625, 0.0013514
5: 0.0027524, 0.0043995, 0.0028981, 0.0043733, -0.0014355, 0.0011921
6: -0.0110747, -0.0048443, -0.0107424, -0.0049484, -0.0051807, 0.0046358
7: 0.0036344, 0.0123958, 0.0040996, 0.0120428, -0.0066919, 0.0069622
8: 0.9920603, 0.9979755, 0.9921602, 0.9977044, -0.0044260, 0.0048784
9: -0.0140226, -0.0085475, -0.0137968, -0.0087437, -0.0044226, 0.0041456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023185, upper bound: 0.0022617
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023712, upper bound: 0.0022763
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062496, 0.0095222, 0.0062234, 0.0094527, -0.0030454, 0.0030791
1: 0.0015436, 0.0026185, 0.0015325, 0.0026036, -0.0010337, 0.0010598
2: 0.0089361, 0.0109046, 0.0089740, 0.0109191, -0.0019270, 0.0019254
3: -0.0049591, -0.0027587, -0.0049003, -0.0027419, -0.0019662, 0.0018976
4: -0.0004146, 0.0013316, -0.0004304, 0.0012679, -0.0013735, 0.0014330
5: 0.0026501, 0.0044480, 0.0026888, 0.0044634, -0.0016206, 0.0015857
6: -0.0111485, -0.0046519, -0.0109363, -0.0045910, -0.0053910, 0.0051864
7: 0.0030722, 0.0123964, 0.0029998, 0.0120708, -0.0074465, 0.0077546
8: 0.9918758, 0.9979973, 0.9918173, 0.9977711, -0.0047890, 0.0049961
9: -0.0140230, -0.0082838, -0.0138147, -0.0082327, -0.0047173, 0.0045227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023956, upper bound: 0.0024327
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024525, upper bound: 0.0024524
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063776, 0.0088980, 0.0062251, 0.0089385, -0.0020515, 0.0022214
1: 0.0021940, 0.0026035, 0.0022216, 0.0026136, -0.0003508, 0.0003146
2: 0.0094350, 0.0108339, 0.0094180, 0.0109182, -0.0012379, 0.0011342
3: -0.0048997, -0.0034268, -0.0049399, -0.0033884, -0.0012450, 0.0012269
4: -0.0002793, 0.0012673, -0.0003689, 0.0013108, -0.0012753, 0.0013478
5: 0.0029015, 0.0043731, 0.0028729, 0.0044624, -0.0012926, 0.0012018
6: -0.0107519, -0.0049493, -0.0109014, -0.0045948, -0.0050791, 0.0047683
7: 0.0041275, 0.0120676, 0.0037010, 0.0122901, -0.0065555, 0.0068921
8: 0.9921610, 0.9977192, 0.9918209, 0.9978712, -0.0045745, 0.0048611
9: -0.0138127, -0.0087533, -0.0139550, -0.0084629, -0.0044070, 0.0041724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0023314
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022770, upper bound: 0.0023395
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063075, 0.0091240, 0.0060662, 0.0094164, -0.0028947, 0.0026367
1: 0.0018924, 0.0026035, 0.0016749, 0.0026176, -0.0006881, 0.0008896
2: 0.0092619, 0.0108726, 0.0090428, 0.0110060, -0.0015537, 0.0017630
3: -0.0048998, -0.0031176, -0.0049556, -0.0028132, -0.0018044, 0.0015986
4: -0.0003479, 0.0012673, -0.0005130, 0.0013278, -0.0013900, 0.0014246
5: 0.0028096, 0.0044141, 0.0026779, 0.0045555, -0.0014631, 0.0015593
6: -0.0108276, -0.0047863, -0.0111300, -0.0042255, -0.0053010, 0.0053065
7: 0.0036116, 0.0120680, 0.0026698, 0.0123772, -0.0073243, 0.0076186
8: 0.9920046, 0.9977425, 0.9914666, 0.9979827, -0.0049482, 0.0049912
9: -0.0138130, -0.0085167, -0.0140107, -0.0079698, -0.0046827, 0.0045609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0025268
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0025452
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063324, 0.0092319, 0.0062249, 0.0089662, -0.0021150, 0.0027620
1: 0.0018506, 0.0026185, 0.0021877, 0.0026136, -0.0007117, 0.0003776
2: 0.0091946, 0.0108588, 0.0093979, 0.0109183, -0.0016223, 0.0011762
3: -0.0049590, -0.0030887, -0.0049399, -0.0033546, -0.0013801, 0.0015694
4: -0.0003378, 0.0013314, -0.0003724, 0.0013107, -0.0013221, 0.0014565
5: 0.0027524, 0.0043995, 0.0028611, 0.0044625, -0.0015392, 0.0012326
6: -0.0110747, -0.0048443, -0.0109144, -0.0045945, -0.0055919, 0.0048446
7: 0.0036344, 0.0123958, 0.0036613, 0.0122898, -0.0069966, 0.0074727
8: 0.9920603, 0.9979755, 0.9918206, 0.9978755, -0.0046366, 0.0052729
9: -0.0140226, -0.0085475, -0.0139548, -0.0084498, -0.0047645, 0.0043404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023185, upper bound: 0.0023384
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023706, upper bound: 0.0023477
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062496, 0.0095222, 0.0060551, 0.0095133, -0.0030628, 0.0032638
1: 0.0015436, 0.0026185, 0.0015816, 0.0026176, -0.0010488, 0.0010127
2: 0.0089361, 0.0109046, 0.0089595, 0.0110122, -0.0020291, 0.0018960
3: -0.0049591, -0.0027587, -0.0049556, -0.0027216, -0.0019986, 0.0019571
4: -0.0004146, 0.0013316, -0.0005284, 0.0013278, -0.0014379, 0.0015421
5: 0.0026501, 0.0044480, 0.0026436, 0.0045620, -0.0017288, 0.0016228
6: -0.0111485, -0.0046519, -0.0111574, -0.0041997, -0.0058202, 0.0053955
7: 0.0030722, 0.0123964, 0.0025389, 0.0123770, -0.0077759, 0.0082756
8: 0.9918758, 0.9979973, 0.9914419, 0.9979908, -0.0050157, 0.0054079
9: -0.0140230, -0.0082838, -0.0140106, -0.0079153, -0.0050715, 0.0047333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023956, upper bound: 0.0025335
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024520, upper bound: 0.0025508
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062145, 0.0089811, 0.0063827, 0.0088602, -0.0021389, 0.0021200
1: 0.0021969, 0.0026174, 0.0022444, 0.0026023, -0.0003348, 0.0003033
2: 0.0093915, 0.0109240, 0.0094613, 0.0108310, -0.0011759, 0.0011825
3: -0.0049548, -0.0033592, -0.0048952, -0.0034785, -0.0012006, 0.0012481
4: -0.0003778, 0.0013269, -0.0002713, 0.0012623, -0.0013265, 0.0012997
5: 0.0028506, 0.0044686, 0.0029187, 0.0043701, -0.0012387, 0.0012529
6: -0.0109700, -0.0045701, -0.0107196, -0.0049612, -0.0048900, 0.0049713
7: 0.0036405, 0.0123726, 0.0042001, 0.0120424, -0.0067995, 0.0066463
8: 0.9917973, 0.9979320, 0.9921725, 0.9976968, -0.0047693, 0.0046851
9: -0.0140077, -0.0084326, -0.0137966, -0.0087820, -0.0042498, 0.0043387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023493, upper bound: 0.0022549
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023832, upper bound: 0.0022684
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061325, 0.0092067, 0.0062416, 0.0093425, -0.0030258, 0.0025399
1: 0.0019043, 0.0026174, 0.0016423, 0.0026036, -0.0006607, 0.0009372
2: 0.0092178, 0.0109694, 0.0090745, 0.0109090, -0.0014937, 0.0018611
3: -0.0049549, -0.0030547, -0.0049003, -0.0028513, -0.0018126, 0.0016147
4: -0.0004528, 0.0013270, -0.0004110, 0.0012679, -0.0014395, 0.0013722
5: 0.0027601, 0.0045167, 0.0027269, 0.0044527, -0.0014107, 0.0016210
6: -0.0110592, -0.0043796, -0.0109032, -0.0046333, -0.0050928, 0.0055278
7: 0.0030983, 0.0123731, 0.0031529, 0.0120707, -0.0075572, 0.0073829
8: 0.9916145, 0.9979593, 0.9918578, 0.9977633, -0.0051406, 0.0047860
9: -0.0140080, -0.0081755, -0.0138147, -0.0083002, -0.0045140, 0.0047212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024263, upper bound: 0.0024250
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024698, upper bound: 0.0024449
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061776, 0.0092813, 0.0063772, 0.0089098, -0.0022402, 0.0026168
1: 0.0018938, 0.0026307, 0.0021676, 0.0026024, -0.0006513, 0.0004116
2: 0.0091757, 0.0109444, 0.0094244, 0.0108341, -0.0015279, 0.0012527
3: -0.0050072, -0.0030674, -0.0048952, -0.0034033, -0.0013785, 0.0015535
4: -0.0004274, 0.0013837, -0.0002819, 0.0012624, -0.0013691, 0.0014137
5: 0.0027153, 0.0044902, 0.0028981, 0.0043733, -0.0014619, 0.0012993
6: -0.0112593, -0.0044845, -0.0107424, -0.0049484, -0.0053876, 0.0050614
7: 0.0032156, 0.0126629, 0.0040996, 0.0120428, -0.0071974, 0.0072806
8: 0.9917152, 0.9981607, 0.9921602, 0.9977044, -0.0048343, 0.0050945
9: -0.0141934, -0.0082575, -0.0137968, -0.0087437, -0.0046263, 0.0044914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024191, upper bound: 0.0022617
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024717, upper bound: 0.0022763
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0060856, 0.0095822, 0.0062234, 0.0094527, -0.0032338, 0.0031266
1: 0.0015851, 0.0026307, 0.0015325, 0.0026036, -0.0009936, 0.0010730
2: 0.0089158, 0.0109953, 0.0089740, 0.0109191, -0.0019214, 0.0020213
3: -0.0050074, -0.0027319, -0.0049003, -0.0027419, -0.0020183, 0.0019378
4: -0.0005097, 0.0013838, -0.0004304, 0.0012679, -0.0014837, 0.0014911
5: 0.0026058, 0.0045441, 0.0026888, 0.0044634, -0.0016662, 0.0016960
6: -0.0113389, -0.0042705, -0.0109363, -0.0045910, -0.0055931, 0.0056242
7: 0.0026313, 0.0126636, 0.0029998, 0.0120708, -0.0079727, 0.0080497
8: 0.9915099, 0.9981849, 0.9918173, 0.9977711, -0.0052091, 0.0052003
9: -0.0141938, -0.0079762, -0.0138147, -0.0082327, -0.0049069, 0.0048801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024933, upper bound: 0.0024318
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025508, upper bound: 0.0024520
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062145, 0.0089811, 0.0062251, 0.0089385, -0.0020198, 0.0020839
1: 0.0021969, 0.0026174, 0.0022216, 0.0026136, -0.0003187, 0.0002982
2: 0.0093915, 0.0109240, 0.0094180, 0.0109182, -0.0011558, 0.0011167
3: -0.0049548, -0.0033592, -0.0049399, -0.0033884, -0.0011804, 0.0011817
4: -0.0003778, 0.0013269, -0.0003689, 0.0013108, -0.0012529, 0.0012779
5: 0.0028506, 0.0044686, 0.0028729, 0.0044624, -0.0012176, 0.0011832
6: -0.0109700, -0.0045701, -0.0109014, -0.0045948, -0.0048074, 0.0046946
7: 0.0036405, 0.0123726, 0.0037010, 0.0122901, -0.0064241, 0.0065347
8: 0.9917973, 0.9979320, 0.9918209, 0.9978712, -0.0045038, 0.0046062
9: -0.0140077, -0.0084326, -0.0139550, -0.0084629, -0.0041785, 0.0040981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0022604
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023833, upper bound: 0.0022684
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061325, 0.0092067, 0.0060662, 0.0094164, -0.0028251, 0.0024695
1: 0.0019043, 0.0026174, 0.0016749, 0.0026176, -0.0006541, 0.0008825
2: 0.0092178, 0.0109694, 0.0090428, 0.0110060, -0.0014507, 0.0017309
3: -0.0049549, -0.0030547, -0.0049556, -0.0028132, -0.0017489, 0.0015552
4: -0.0004528, 0.0013270, -0.0005130, 0.0013278, -0.0013685, 0.0013488
5: 0.0027601, 0.0045167, 0.0026779, 0.0045555, -0.0013745, 0.0015208
6: -0.0110592, -0.0043796, -0.0111300, -0.0042255, -0.0050045, 0.0052364
7: 0.0030983, 0.0123731, 0.0026698, 0.0123772, -0.0071993, 0.0072377
8: 0.9916145, 0.9979593, 0.9914666, 0.9979827, -0.0048761, 0.0047110
9: -0.0140080, -0.0081755, -0.0140107, -0.0079698, -0.0044355, 0.0044896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024332, upper bound: 0.0024403
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024743, upper bound: 0.0024580
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061776, 0.0092813, 0.0062249, 0.0089662, -0.0020801, 0.0025724
1: 0.0018938, 0.0026307, 0.0021877, 0.0026136, -0.0006427, 0.0003619
2: 0.0091757, 0.0109444, 0.0093979, 0.0109183, -0.0014951, 0.0011562
3: -0.0050072, -0.0030674, -0.0049399, -0.0033546, -0.0013161, 0.0014936
4: -0.0004274, 0.0013837, -0.0003724, 0.0013107, -0.0012975, 0.0013865
5: 0.0027153, 0.0044902, 0.0028611, 0.0044625, -0.0014429, 0.0012128
6: -0.0112593, -0.0044845, -0.0109144, -0.0045945, -0.0053097, 0.0047694
7: 0.0032156, 0.0126629, 0.0036613, 0.0122898, -0.0068363, 0.0071152
8: 0.9917152, 0.9981607, 0.9918206, 0.9978755, -0.0045653, 0.0050151
9: -0.0141934, -0.0082575, -0.0139548, -0.0084498, -0.0045355, 0.0042575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024242, upper bound: 0.0022716
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024729, upper bound: 0.0022790
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0060856, 0.0095822, 0.0060551, 0.0095133, -0.0030044, 0.0030472
1: 0.0015851, 0.0026307, 0.0015816, 0.0026176, -0.0009864, 0.0010042
2: 0.0089158, 0.0109953, 0.0089595, 0.0110122, -0.0018885, 0.0018739
3: -0.0050074, -0.0027319, -0.0049556, -0.0027216, -0.0019423, 0.0018872
4: -0.0005097, 0.0013838, -0.0005284, 0.0013278, -0.0014142, 0.0014668
5: 0.0026058, 0.0045441, 0.0026436, 0.0045620, -0.0016235, 0.0015865
6: -0.0113389, -0.0042705, -0.0111574, -0.0041997, -0.0055149, 0.0053276
7: 0.0026313, 0.0126636, 0.0025389, 0.0123770, -0.0076249, 0.0078943
8: 0.9915099, 0.9981849, 0.9914419, 0.9979908, -0.0049448, 0.0051299
9: -0.0141938, -0.0079762, -0.0140106, -0.0079153, -0.0048253, 0.0046533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024992, upper bound: 0.0024468
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025573, upper bound: 0.0024643
time: 0.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022549
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0022773, upper bound: 0.0022684
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0024261
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0024458
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023185, upper bound: 0.0022617
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023712, upper bound: 0.0022763
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023956, upper bound: 0.0024327
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024525, upper bound: 0.0024524
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0023314
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0022770, upper bound: 0.0023395
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0025268
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0025452
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023185, upper bound: 0.0023384
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023706, upper bound: 0.0023477
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023956, upper bound: 0.0025335
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024520, upper bound: 0.0025508
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023493, upper bound: 0.0022549
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023832, upper bound: 0.0022684
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024263, upper bound: 0.0024250
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024698, upper bound: 0.0024449
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024191, upper bound: 0.0022617
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024717, upper bound: 0.0022763
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024933, upper bound: 0.0024318
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0025508, upper bound: 0.0024520
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0022604
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0023833, upper bound: 0.0022684
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024332, upper bound: 0.0024403
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024743, upper bound: 0.0024580
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024242, upper bound: 0.0022716
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024729, upper bound: 0.0022790
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0024992, upper bound: 0.0024468
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 8, lower bound: -0.0025573, upper bound: 0.0024643

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0064653, 0.0088359, 0.0064247, 0.0088513, -0.0018399, 0.0018915
1: 0.0022563, 0.0025988, 0.0022505, 0.0026011, -0.0002658, 0.0002733
2: 0.0094747, 0.0107854, 0.0094662, 0.0108078, -0.0010458, 0.0010172
3: -0.0048812, -0.0035257, -0.0048901, -0.0035025, -0.0010816, 0.0010521
4: -0.0002202, 0.0012473, -0.0002453, 0.0012568, -0.0011389, 0.0011709
5: 0.0029330, 0.0043217, 0.0029240, 0.0043455, -0.0011080, 0.0010778
6: -0.0106630, -0.0051531, -0.0106989, -0.0050587, -0.0043964, 0.0042764
7: 0.0044614, 0.0119653, 0.0043328, 0.0120143, -0.0058240, 0.0059875
8: 0.9923565, 0.9976425, 0.9922661, 0.9976770, -0.0041026, 0.0042177
9: -0.0137473, -0.0089491, -0.0137786, -0.0088669, -0.0038286, 0.0037240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022225
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022549
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064343, 0.0088552, 0.0064077, 0.0088553, -0.0018416, 0.0019680
1: 0.0022519, 0.0026016, 0.0022480, 0.0026016, -0.0002661, 0.0002843
2: 0.0094641, 0.0108025, 0.0094640, 0.0108172, -0.0010881, 0.0010182
3: -0.0048923, -0.0035080, -0.0048924, -0.0034928, -0.0011253, 0.0010530
4: -0.0002393, 0.0012592, -0.0002558, 0.0012593, -0.0011400, 0.0012182
5: 0.0029217, 0.0043398, 0.0029216, 0.0043554, -0.0011529, 0.0010788
6: -0.0107079, -0.0050811, -0.0107082, -0.0050193, -0.0045742, 0.0042804
7: 0.0043633, 0.0120266, 0.0042791, 0.0120270, -0.0058295, 0.0062297
8: 0.9922875, 0.9976856, 0.9922282, 0.9976859, -0.0041064, 0.0043883
9: -0.0137865, -0.0088864, -0.0137867, -0.0088325, -0.0039834, 0.0037275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022773, upper bound: 0.0022360
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022773, upper bound: 0.0022684
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0064040, 0.0089179, 0.0062856, 0.0092282, -0.0025284, 0.0020436
1: 0.0021298, 0.0025988, 0.0017573, 0.0026023, -0.0004089, 0.0007895
2: 0.0094137, 0.0108192, 0.0091751, 0.0108847, -0.0011548, 0.0015349
3: -0.0048812, -0.0033818, -0.0048949, -0.0029788, -0.0015786, 0.0012401
4: -0.0002688, 0.0012473, -0.0003740, 0.0012621, -0.0012255, 0.0012349
5: 0.0028993, 0.0043576, 0.0027717, 0.0044270, -0.0011751, 0.0013668
6: -0.0106946, -0.0050108, -0.0108467, -0.0047355, -0.0045088, 0.0047468
7: 0.0041444, 0.0119654, 0.0034003, 0.0120411, -0.0063415, 0.0066095
8: 0.9922200, 0.9976521, 0.9919559, 0.9977336, -0.0044238, 0.0042975
9: -0.0137473, -0.0087849, -0.0137957, -0.0084255, -0.0040599, 0.0040128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0023523
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0024261
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063700, 0.0090073, 0.0062690, 0.0092807, -0.0026428, 0.0022273
1: 0.0020317, 0.0026016, 0.0017027, 0.0026028, -0.0005148, 0.0008579
2: 0.0093507, 0.0108381, 0.0091283, 0.0108939, -0.0012803, 0.0016257
3: -0.0048923, -0.0032743, -0.0048970, -0.0029206, -0.0016829, 0.0013408
4: -0.0002985, 0.0012593, -0.0003891, 0.0012643, -0.0012406, 0.0012888
5: 0.0028601, 0.0043775, 0.0027508, 0.0044367, -0.0012628, 0.0014124
6: -0.0107636, -0.0049316, -0.0108718, -0.0046970, -0.0047344, 0.0047889
7: 0.0039353, 0.0120267, 0.0032954, 0.0120526, -0.0064776, 0.0069231
8: 0.9921440, 0.9977032, 0.9919190, 0.9977459, -0.0044533, 0.0044873
9: -0.0137866, -0.0086834, -0.0138031, -0.0083741, -0.0042388, 0.0040664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0023717
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0024458
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0064264, 0.0090096, 0.0064195, 0.0088514, -0.0018660, 0.0021784
1: 0.0021116, 0.0026122, 0.0022497, 0.0026011, -0.0004140, 0.0002969
2: 0.0093637, 0.0108069, 0.0094662, 0.0108107, -0.0012311, 0.0010316
3: -0.0049342, -0.0033694, -0.0048901, -0.0034995, -0.0011750, 0.0012023
4: -0.0002576, 0.0013046, -0.0002485, 0.0012569, -0.0011683, 0.0012720
5: 0.0028450, 0.0043445, 0.0029239, 0.0043485, -0.0012533, 0.0010931
6: -0.0109112, -0.0050628, -0.0106991, -0.0050467, -0.0048185, 0.0043370
7: 0.0041839, 0.0122585, 0.0043165, 0.0120145, -0.0060601, 0.0065045
8: 0.9922699, 0.9978596, 0.9922545, 0.9976772, -0.0041608, 0.0045958
9: -0.0139348, -0.0088202, -0.0137788, -0.0088564, -0.0041592, 0.0038266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023092, upper bound: 0.0022225
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023092, upper bound: 0.0022337
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063869, 0.0091103, 0.0064018, 0.0088714, -0.0018898, 0.0023769
1: 0.0019941, 0.0026166, 0.0022220, 0.0026017, -0.0005351, 0.0003366
2: 0.0092884, 0.0108287, 0.0094524, 0.0108205, -0.0013641, 0.0010492
3: -0.0049515, -0.0032459, -0.0048924, -0.0034642, -0.0012499, 0.0013124
4: -0.0002918, 0.0013233, -0.0002621, 0.0012594, -0.0011798, 0.0013258
5: 0.0028010, 0.0043676, 0.0029147, 0.0043589, -0.0013470, 0.0011033
6: -0.0110097, -0.0049708, -0.0107162, -0.0050054, -0.0050503, 0.0043501
7: 0.0039447, 0.0123542, 0.0042309, 0.0120272, -0.0061863, 0.0067977
8: 0.9921817, 0.9979361, 0.9922149, 0.9976888, -0.0041663, 0.0047920
9: -0.0139960, -0.0087035, -0.0137869, -0.0088109, -0.0043365, 0.0038691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023627, upper bound: 0.0022361
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023627, upper bound: 0.0022485
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063535, 0.0092428, 0.0062696, 0.0093246, -0.0027205, 0.0025766
1: 0.0018053, 0.0026122, 0.0016506, 0.0026023, -0.0007504, 0.0009252
2: 0.0091731, 0.0108472, 0.0090868, 0.0108936, -0.0015445, 0.0016922
3: -0.0049342, -0.0030505, -0.0048949, -0.0028695, -0.0017829, 0.0015690
4: -0.0003296, 0.0013046, -0.0003928, 0.0012621, -0.0012740, 0.0013495
5: 0.0027548, 0.0043872, 0.0027359, 0.0044363, -0.0014150, 0.0014437
6: -0.0109878, -0.0048933, -0.0108795, -0.0046983, -0.0050235, 0.0048570
7: 0.0036435, 0.0122588, 0.0032497, 0.0120412, -0.0067804, 0.0072588
8: 0.9921073, 0.9978809, 0.9919202, 0.9977416, -0.0045061, 0.0047089
9: -0.0139349, -0.0085719, -0.0137958, -0.0083602, -0.0044395, 0.0041855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0023470
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0023601
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063077, 0.0093598, 0.0062495, 0.0093808, -0.0028486, 0.0027908
1: 0.0016922, 0.0026166, 0.0015961, 0.0026028, -0.0008672, 0.0009928
2: 0.0090791, 0.0108725, 0.0090375, 0.0109046, -0.0017015, 0.0017922
3: -0.0049515, -0.0029236, -0.0048970, -0.0028106, -0.0018868, 0.0016882
4: -0.0003664, 0.0013234, -0.0004095, 0.0012643, -0.0012904, 0.0014055
5: 0.0027086, 0.0044140, 0.0027147, 0.0044481, -0.0015087, 0.0014908
6: -0.0110823, -0.0047868, -0.0109041, -0.0046517, -0.0052507, 0.0048962
7: 0.0033997, 0.0123546, 0.0031373, 0.0120526, -0.0069334, 0.0075765
8: 0.9920052, 0.9979572, 0.9918755, 0.9977536, -0.0045333, 0.0049012
9: -0.0139962, -0.0084475, -0.0138031, -0.0083033, -0.0046246, 0.0042442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024458, upper bound: 0.0023663
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024458, upper bound: 0.0023793
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0064653, 0.0088359, 0.0062651, 0.0089301, -0.0019355, 0.0020706
1: 0.0022563, 0.0025988, 0.0022274, 0.0026124, -0.0002796, 0.0002991
2: 0.0094747, 0.0107854, 0.0094226, 0.0108961, -0.0011448, 0.0010701
3: -0.0048812, -0.0035257, -0.0049351, -0.0034112, -0.0011840, 0.0011067
4: -0.0002202, 0.0012473, -0.0003441, 0.0013056, -0.0011981, 0.0012817
5: 0.0029330, 0.0043217, 0.0028778, 0.0044390, -0.0012129, 0.0011338
6: -0.0106630, -0.0051531, -0.0108820, -0.0046878, -0.0048126, 0.0044986
7: 0.0044614, 0.0119653, 0.0038276, 0.0122636, -0.0061267, 0.0065543
8: 0.9923565, 0.9976425, 0.9919101, 0.9978526, -0.0043158, 0.0046170
9: -0.0137473, -0.0089491, -0.0139380, -0.0085438, -0.0041910, 0.0039176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022883
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0023314
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064343, 0.0088552, 0.0062520, 0.0089334, -0.0019457, 0.0021449
1: 0.0022519, 0.0026016, 0.0022255, 0.0026129, -0.0002811, 0.0003099
2: 0.0094641, 0.0108025, 0.0094208, 0.0109033, -0.0011859, 0.0010757
3: -0.0048923, -0.0035080, -0.0049370, -0.0034037, -0.0012265, 0.0011126
4: -0.0002393, 0.0012592, -0.0003522, 0.0013076, -0.0012044, 0.0013277
5: 0.0029217, 0.0043398, 0.0028759, 0.0044467, -0.0012565, 0.0011398
6: -0.0107079, -0.0050811, -0.0108896, -0.0046573, -0.0049854, 0.0045223
7: 0.0043633, 0.0120266, 0.0037861, 0.0122740, -0.0061590, 0.0067897
8: 0.9922875, 0.9976856, 0.9918809, 0.9978600, -0.0043386, 0.0047828
9: -0.0137865, -0.0088864, -0.0139447, -0.0085173, -0.0043415, 0.0039383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0022959
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0023395
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0064040, 0.0089179, 0.0061123, 0.0093025, -0.0025944, 0.0022282
1: 0.0021298, 0.0025988, 0.0017871, 0.0026164, -0.0004241, 0.0007578
2: 0.0094137, 0.0108192, 0.0091411, 0.0109805, -0.0012569, 0.0015529
3: -0.0048812, -0.0033818, -0.0049509, -0.0029357, -0.0016293, 0.0013003
4: -0.0002688, 0.0012473, -0.0004753, 0.0013227, -0.0012907, 0.0013427
5: 0.0028993, 0.0043576, 0.0027252, 0.0045285, -0.0012833, 0.0014174
6: -0.0106946, -0.0050108, -0.0110757, -0.0043326, -0.0049379, 0.0049659
7: 0.0041444, 0.0119654, 0.0029173, 0.0123510, -0.0066748, 0.0071284
8: 0.9922200, 0.9976521, 0.9915695, 0.9979537, -0.0046548, 0.0047092
9: -0.0137473, -0.0087849, -0.0139939, -0.0080971, -0.0044096, 0.0042258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0024548
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0025268
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063700, 0.0090073, 0.0060933, 0.0093527, -0.0026962, 0.0024124
1: 0.0020317, 0.0026016, 0.0017363, 0.0026169, -0.0005300, 0.0008220
2: 0.0093507, 0.0108381, 0.0090982, 0.0109910, -0.0013827, 0.0016275
3: -0.0048923, -0.0032743, -0.0049526, -0.0028824, -0.0017256, 0.0014012
4: -0.0002985, 0.0012593, -0.0004914, 0.0013245, -0.0013060, 0.0013982
5: 0.0028601, 0.0043775, 0.0027045, 0.0045396, -0.0013712, 0.0014568
6: -0.0107636, -0.0049316, -0.0110980, -0.0042886, -0.0051647, 0.0050289
7: 0.0039353, 0.0120267, 0.0028122, 0.0123603, -0.0068119, 0.0074474
8: 0.9921440, 0.9977032, 0.9915271, 0.9979649, -0.0046877, 0.0049001
9: -0.0137866, -0.0086834, -0.0139998, -0.0080426, -0.0045938, 0.0042801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0024720
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0025452
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0064264, 0.0090096, 0.0062673, 0.0089300, -0.0019621, 0.0023544
1: 0.0021116, 0.0026122, 0.0022277, 0.0026124, -0.0004278, 0.0003223
2: 0.0093637, 0.0108069, 0.0094227, 0.0108948, -0.0013284, 0.0010848
3: -0.0049342, -0.0033694, -0.0049350, -0.0034125, -0.0012756, 0.0012576
4: -0.0002576, 0.0013046, -0.0003427, 0.0013055, -0.0012278, 0.0013809
5: 0.0028450, 0.0043445, 0.0028779, 0.0044377, -0.0013563, 0.0011494
6: -0.0109112, -0.0050628, -0.0108817, -0.0046929, -0.0052274, 0.0045604
7: 0.0041839, 0.0122585, 0.0038347, 0.0122632, -0.0063640, 0.0070614
8: 0.9922699, 0.9978596, 0.9919151, 0.9978523, -0.0043751, 0.0049881
9: -0.0139348, -0.0088202, -0.0139378, -0.0085483, -0.0045153, 0.0040210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023093, upper bound: 0.0022864
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023092, upper bound: 0.0022986
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063869, 0.0091103, 0.0062502, 0.0089333, -0.0019712, 0.0025537
1: 0.0019941, 0.0026166, 0.0022253, 0.0026129, -0.0005502, 0.0003343
2: 0.0092884, 0.0108287, 0.0094209, 0.0109043, -0.0014618, 0.0010898
3: -0.0049515, -0.0032459, -0.0049369, -0.0034027, -0.0013232, 0.0013722
4: -0.0002918, 0.0013233, -0.0003533, 0.0013076, -0.0012443, 0.0014324
5: 0.0028010, 0.0043676, 0.0028760, 0.0044477, -0.0014505, 0.0011547
6: -0.0110097, -0.0049708, -0.0108894, -0.0046533, -0.0054611, 0.0045817
7: 0.0039447, 0.0123542, 0.0037806, 0.0122737, -0.0065157, 0.0073248
8: 0.9921817, 0.9979361, 0.9918770, 0.9978597, -0.0043955, 0.0051861
9: -0.0139960, -0.0087035, -0.0139445, -0.0085138, -0.0046837, 0.0040801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023622, upper bound: 0.0022945
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023622, upper bound: 0.0023081
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063535, 0.0092428, 0.0061013, 0.0093859, -0.0027455, 0.0027601
1: 0.0018053, 0.0026122, 0.0016956, 0.0026164, -0.0007649, 0.0008791
2: 0.0091731, 0.0108472, 0.0090661, 0.0109866, -0.0016459, 0.0016766
3: -0.0049342, -0.0030505, -0.0049508, -0.0028470, -0.0018195, 0.0016257
4: -0.0003296, 0.0013046, -0.0004908, 0.0013226, -0.0013360, 0.0014574
5: 0.0027548, 0.0043872, 0.0026918, 0.0045349, -0.0015224, 0.0014798
6: -0.0109878, -0.0048933, -0.0111019, -0.0043071, -0.0054499, 0.0050563
7: 0.0036435, 0.0122588, 0.0027882, 0.0123507, -0.0070975, 0.0077762
8: 0.9921073, 0.9978809, 0.9915449, 0.9979613, -0.0047234, 0.0051179
9: -0.0139349, -0.0085719, -0.0139937, -0.0080426, -0.0047894, 0.0043882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0024510
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0024631
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063077, 0.0093598, 0.0060821, 0.0094412, -0.0028558, 0.0029754
1: 0.0016922, 0.0026166, 0.0016433, 0.0026168, -0.0008818, 0.0009447
2: 0.0090791, 0.0108725, 0.0090197, 0.0109972, -0.0018035, 0.0017590
3: -0.0049515, -0.0029236, -0.0049525, -0.0027898, -0.0019180, 0.0017453
4: -0.0003664, 0.0013234, -0.0005070, 0.0013244, -0.0013528, 0.0015146
5: 0.0027086, 0.0044140, 0.0026704, 0.0045462, -0.0016168, 0.0015243
6: -0.0110823, -0.0047868, -0.0111249, -0.0042625, -0.0056798, 0.0051207
7: 0.0033997, 0.0123546, 0.0026785, 0.0123600, -0.0072534, 0.0080992
8: 0.9920052, 0.9979572, 0.9915022, 0.9979727, -0.0047550, 0.0053129
9: -0.0139962, -0.0084475, -0.0139997, -0.0079876, -0.0049787, 0.0044483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024698
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024813
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063038, 0.0089297, 0.0064247, 0.0088513, -0.0020305, 0.0019948
1: 0.0022330, 0.0026124, 0.0022505, 0.0026011, -0.0002933, 0.0002882
2: 0.0094229, 0.0108747, 0.0094662, 0.0108078, -0.0011029, 0.0011226
3: -0.0049349, -0.0034334, -0.0048901, -0.0035025, -0.0011406, 0.0011611
4: -0.0003202, 0.0013053, -0.0002453, 0.0012568, -0.0012569, 0.0012348
5: 0.0028781, 0.0044163, 0.0029240, 0.0043455, -0.0011685, 0.0011895
6: -0.0108810, -0.0047777, -0.0106989, -0.0050587, -0.0046364, 0.0047195
7: 0.0039501, 0.0122622, 0.0043328, 0.0120143, -0.0064275, 0.0063144
8: 0.9919963, 0.9978516, 0.9922661, 0.9976770, -0.0045277, 0.0044480
9: -0.0139372, -0.0086221, -0.0137786, -0.0088669, -0.0040376, 0.0041099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023493, upper bound: 0.0022225
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023493, upper bound: 0.0022549
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062742, 0.0089521, 0.0064077, 0.0088553, -0.0020264, 0.0020684
1: 0.0022287, 0.0026156, 0.0022480, 0.0026016, -0.0002928, 0.0002988
2: 0.0094105, 0.0108910, 0.0094640, 0.0108172, -0.0011435, 0.0011203
3: -0.0049477, -0.0034164, -0.0048924, -0.0034928, -0.0011827, 0.0011587
4: -0.0003385, 0.0013192, -0.0002558, 0.0012593, -0.0012544, 0.0012803
5: 0.0028649, 0.0044337, 0.0029216, 0.0043554, -0.0012116, 0.0011870
6: -0.0109331, -0.0047089, -0.0107082, -0.0050193, -0.0048074, 0.0047098
7: 0.0038564, 0.0123333, 0.0042791, 0.0120270, -0.0064144, 0.0065473
8: 0.9919304, 0.9979017, 0.9922282, 0.9976859, -0.0045184, 0.0046121
9: -0.0139826, -0.0085622, -0.0137867, -0.0088325, -0.0041865, 0.0041015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023832, upper bound: 0.0022361
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023832, upper bound: 0.0022684
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062336, 0.0089878, 0.0062856, 0.0092282, -0.0027208, 0.0021176
1: 0.0021440, 0.0026124, 0.0017573, 0.0026023, -0.0003956, 0.0008043
2: 0.0093805, 0.0109134, 0.0091751, 0.0108847, -0.0011878, 0.0016413
3: -0.0049349, -0.0033184, -0.0048949, -0.0029788, -0.0016368, 0.0013124
4: -0.0003711, 0.0013053, -0.0003740, 0.0012621, -0.0013410, 0.0012986
5: 0.0028538, 0.0044574, 0.0027717, 0.0044270, -0.0012248, 0.0014795
6: -0.0109073, -0.0046147, -0.0108467, -0.0047355, -0.0047429, 0.0051941
7: 0.0036416, 0.0122623, 0.0034003, 0.0120411, -0.0069088, 0.0069357
8: 0.9918400, 0.9978604, 0.9919559, 0.9977336, -0.0048529, 0.0045263
9: -0.0139372, -0.0084520, -0.0137957, -0.0084255, -0.0042681, 0.0043887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024263, upper bound: 0.0023523
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024263, upper bound: 0.0024250
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061939, 0.0090750, 0.0062690, 0.0092807, -0.0028319, 0.0022952
1: 0.0020511, 0.0026156, 0.0017027, 0.0026028, -0.0004931, 0.0008710
2: 0.0093184, 0.0109354, 0.0091283, 0.0108939, -0.0013102, 0.0017303
3: -0.0049477, -0.0032169, -0.0048970, -0.0029206, -0.0017344, 0.0014066
4: -0.0004035, 0.0013193, -0.0003891, 0.0012643, -0.0013537, 0.0013455
5: 0.0028141, 0.0044807, 0.0027508, 0.0044367, -0.0013070, 0.0015232
6: -0.0109852, -0.0045223, -0.0108718, -0.0046970, -0.0049420, 0.0052286
7: 0.0034260, 0.0123335, 0.0032954, 0.0120526, -0.0070283, 0.0072124
8: 0.9917513, 0.9979182, 0.9919190, 0.9977459, -0.0048751, 0.0046896
9: -0.0139827, -0.0083422, -0.0138031, -0.0083741, -0.0044238, 0.0044341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024698, upper bound: 0.0023717
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024698, upper bound: 0.0024449
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062713, 0.0090630, 0.0064195, 0.0088514, -0.0020541, 0.0022268
1: 0.0021509, 0.0026245, 0.0022497, 0.0026011, -0.0003771, 0.0003118
2: 0.0093401, 0.0108927, 0.0094662, 0.0108107, -0.0012442, 0.0011357
3: -0.0049828, -0.0033385, -0.0048901, -0.0034995, -0.0012340, 0.0012491
4: -0.0003480, 0.0013572, -0.0002485, 0.0012569, -0.0012790, 0.0013359
5: 0.0028084, 0.0044354, 0.0029239, 0.0043485, -0.0012922, 0.0012033
6: -0.0110978, -0.0047021, -0.0106991, -0.0050467, -0.0050421, 0.0047744
7: 0.0037583, 0.0125275, 0.0043165, 0.0120145, -0.0065878, 0.0068312
8: 0.9919239, 0.9980459, 0.9922545, 0.9976772, -0.0045803, 0.0048207
9: -0.0141068, -0.0085273, -0.0137788, -0.0088564, -0.0043680, 0.0041857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024105, upper bound: 0.0022225
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024105, upper bound: 0.0022336
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062344, 0.0091598, 0.0064018, 0.0088714, -0.0020732, 0.0024030
1: 0.0020382, 0.0026287, 0.0022220, 0.0026017, -0.0004898, 0.0003510
2: 0.0092712, 0.0109130, 0.0094524, 0.0108205, -0.0013669, 0.0011506
3: -0.0049995, -0.0032212, -0.0048924, -0.0034642, -0.0013071, 0.0013535
4: -0.0003801, 0.0013753, -0.0002621, 0.0012594, -0.0012863, 0.0013878
5: 0.0027642, 0.0044569, 0.0029147, 0.0043589, -0.0013759, 0.0012107
6: -0.0111929, -0.0046165, -0.0107162, -0.0050054, -0.0052583, 0.0047763
7: 0.0035344, 0.0126202, 0.0042309, 0.0120272, -0.0066867, 0.0071146
8: 0.9918417, 0.9981200, 0.9922149, 0.9976888, -0.0045753, 0.0050069
9: -0.0141661, -0.0084179, -0.0137869, -0.0088109, -0.0045391, 0.0042140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024622, upper bound: 0.0022361
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024622, upper bound: 0.0022484
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061898, 0.0092971, 0.0062696, 0.0093246, -0.0029125, 0.0026321
1: 0.0018466, 0.0026245, 0.0016506, 0.0026023, -0.0007085, 0.0009389
2: 0.0091548, 0.0109377, 0.0090868, 0.0108936, -0.0015537, 0.0017983
3: -0.0049828, -0.0030242, -0.0048949, -0.0028695, -0.0018368, 0.0016140
4: -0.0004245, 0.0013573, -0.0003928, 0.0012621, -0.0013864, 0.0014095
5: 0.0027190, 0.0044831, 0.0027359, 0.0044363, -0.0014544, 0.0015562
6: -0.0111788, -0.0045129, -0.0108795, -0.0046983, -0.0052292, 0.0053033
7: 0.0032011, 0.0125278, 0.0032497, 0.0120412, -0.0073098, 0.0075656
8: 0.9917424, 0.9980700, 0.9919202, 0.9977416, -0.0049342, 0.0049200
9: -0.0141070, -0.0082648, -0.0137958, -0.0083602, -0.0046355, 0.0045497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024866, upper bound: 0.0023470
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024866, upper bound: 0.0023601
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061467, 0.0094144, 0.0062495, 0.0093808, -0.0030380, 0.0028428
1: 0.0017325, 0.0026287, 0.0015961, 0.0026028, -0.0008258, 0.0010059
2: 0.0090599, 0.0109615, 0.0090375, 0.0109046, -0.0017070, 0.0018969
3: -0.0049996, -0.0028977, -0.0048970, -0.0028106, -0.0019387, 0.0017304
4: -0.0004607, 0.0013754, -0.0004095, 0.0012643, -0.0014001, 0.0014634
5: 0.0026696, 0.0045083, 0.0027147, 0.0044481, -0.0015504, 0.0016018
6: -0.0112674, -0.0044127, -0.0109041, -0.0046517, -0.0054494, 0.0053365
7: 0.0029564, 0.0126207, 0.0031373, 0.0120526, -0.0074543, 0.0078699
8: 0.9916462, 0.9981434, 0.9918755, 0.9977536, -0.0049557, 0.0051039
9: -0.0141664, -0.0081420, -0.0138031, -0.0083033, -0.0048137, 0.0045994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025452, upper bound: 0.0023663
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025452, upper bound: 0.0023793
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063038, 0.0089297, 0.0062651, 0.0089301, -0.0019030, 0.0019557
1: 0.0022330, 0.0026124, 0.0022274, 0.0026124, -0.0002749, 0.0002825
2: 0.0094229, 0.0108747, 0.0094226, 0.0108961, -0.0010813, 0.0010521
3: -0.0049349, -0.0034334, -0.0049351, -0.0034112, -0.0011183, 0.0010882
4: -0.0003202, 0.0013053, -0.0003441, 0.0013056, -0.0011780, 0.0012106
5: 0.0028781, 0.0044163, 0.0028778, 0.0044390, -0.0011457, 0.0011148
6: -0.0108810, -0.0047777, -0.0108820, -0.0046878, -0.0045457, 0.0044232
7: 0.0039501, 0.0122622, 0.0038276, 0.0122636, -0.0060240, 0.0061908
8: 0.9919963, 0.9978516, 0.9919101, 0.9978526, -0.0042434, 0.0043609
9: -0.0139372, -0.0086221, -0.0139380, -0.0085438, -0.0039586, 0.0038519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0022284
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0022604
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062742, 0.0089521, 0.0062520, 0.0089334, -0.0019036, 0.0020318
1: 0.0022287, 0.0026156, 0.0022255, 0.0026129, -0.0002750, 0.0002935
2: 0.0094105, 0.0108910, 0.0094208, 0.0109033, -0.0011233, 0.0010524
3: -0.0049477, -0.0034164, -0.0049370, -0.0034037, -0.0011618, 0.0010885
4: -0.0003385, 0.0013192, -0.0003522, 0.0013076, -0.0011783, 0.0012577
5: 0.0028649, 0.0044337, 0.0028759, 0.0044467, -0.0011902, 0.0011151
6: -0.0109331, -0.0047089, -0.0108896, -0.0046573, -0.0047224, 0.0044245
7: 0.0038564, 0.0123333, 0.0037861, 0.0122740, -0.0060257, 0.0064315
8: 0.9919304, 0.9979017, 0.9918809, 0.9978600, -0.0042446, 0.0045305
9: -0.0139826, -0.0085622, -0.0139447, -0.0085173, -0.0041125, 0.0038530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023833, upper bound: 0.0022353
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023833, upper bound: 0.0022684
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062336, 0.0089878, 0.0061123, 0.0093025, -0.0025264, 0.0020627
1: 0.0021440, 0.0026124, 0.0017871, 0.0026164, -0.0003812, 0.0007496
2: 0.0093805, 0.0109134, 0.0091411, 0.0109805, -0.0011563, 0.0015147
3: -0.0049349, -0.0033184, -0.0049509, -0.0029357, -0.0015682, 0.0012468
4: -0.0003711, 0.0013053, -0.0004753, 0.0013227, -0.0012670, 0.0012671
5: 0.0028538, 0.0044574, 0.0027252, 0.0045285, -0.0011937, 0.0013811
6: -0.0109073, -0.0046147, -0.0110757, -0.0043326, -0.0046318, 0.0048943
7: 0.0036416, 0.0122623, 0.0029173, 0.0123510, -0.0065327, 0.0067492
8: 0.9918400, 0.9978604, 0.9915695, 0.9979537, -0.0045799, 0.0044239
9: -0.0139372, -0.0084520, -0.0139939, -0.0080971, -0.0041633, 0.0041469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024332, upper bound: 0.0023644
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024332, upper bound: 0.0024403
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061939, 0.0090750, 0.0060933, 0.0093527, -0.0026318, 0.0022397
1: 0.0020511, 0.0026156, 0.0017363, 0.0026169, -0.0004832, 0.0008142
2: 0.0093184, 0.0109354, 0.0090982, 0.0109910, -0.0012756, 0.0015985
3: -0.0049477, -0.0032169, -0.0049526, -0.0028824, -0.0016683, 0.0013450
4: -0.0004035, 0.0013193, -0.0004914, 0.0013245, -0.0012821, 0.0013218
5: 0.0028141, 0.0044807, 0.0027045, 0.0045396, -0.0012809, 0.0014223
6: -0.0109852, -0.0045223, -0.0110980, -0.0042886, -0.0048609, 0.0049351
7: 0.0034260, 0.0123335, 0.0028122, 0.0123603, -0.0066664, 0.0070634
8: 0.9917513, 0.9979182, 0.9915271, 0.9979649, -0.0046104, 0.0046166
9: -0.0139827, -0.0083422, -0.0139998, -0.0080426, -0.0043444, 0.0042005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024743, upper bound: 0.0023822
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024743, upper bound: 0.0024579
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062713, 0.0090630, 0.0062673, 0.0089300, -0.0019282, 0.0021858
1: 0.0021509, 0.0026245, 0.0022277, 0.0026124, -0.0003620, 0.0003059
2: 0.0093401, 0.0108927, 0.0094227, 0.0108948, -0.0012218, 0.0010660
3: -0.0049828, -0.0033385, -0.0049350, -0.0034125, -0.0012108, 0.0011827
4: -0.0003480, 0.0013572, -0.0003427, 0.0013055, -0.0012014, 0.0013108
5: 0.0028084, 0.0044354, 0.0028779, 0.0044377, -0.0012682, 0.0011295
6: -0.0110978, -0.0047021, -0.0108817, -0.0046929, -0.0049473, 0.0044816
7: 0.0037583, 0.0125275, 0.0038347, 0.0122632, -0.0061949, 0.0067030
8: 0.9919239, 0.9980459, 0.9919151, 0.9978523, -0.0042995, 0.0047301
9: -0.0141068, -0.0085273, -0.0139378, -0.0085483, -0.0042861, 0.0039323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024131, upper bound: 0.0022282
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024131, upper bound: 0.0022428
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062344, 0.0091598, 0.0062502, 0.0089333, -0.0019295, 0.0023724
1: 0.0020382, 0.0026287, 0.0022253, 0.0026129, -0.0004786, 0.0003178
2: 0.0092712, 0.0109130, 0.0094209, 0.0109043, -0.0013478, 0.0010668
3: -0.0049995, -0.0032212, -0.0049369, -0.0034027, -0.0012580, 0.0012912
4: -0.0003801, 0.0013753, -0.0003533, 0.0013076, -0.0012130, 0.0013619
5: 0.0027642, 0.0044569, 0.0028760, 0.0044477, -0.0013600, 0.0011303
6: -0.0111929, -0.0046165, -0.0108894, -0.0046533, -0.0051776, 0.0044846
7: 0.0035344, 0.0126202, 0.0037806, 0.0122737, -0.0063210, 0.0069642
8: 0.9918417, 0.9981200, 0.9918770, 0.9978597, -0.0043024, 0.0049260
9: -0.0141661, -0.0084179, -0.0139445, -0.0085138, -0.0044531, 0.0039751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024624, upper bound: 0.0022352
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024624, upper bound: 0.0022506
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061898, 0.0092971, 0.0061013, 0.0093859, -0.0026817, 0.0025560
1: 0.0018466, 0.0026245, 0.0016956, 0.0026164, -0.0006994, 0.0008700
2: 0.0091548, 0.0109377, 0.0090661, 0.0109866, -0.0015099, 0.0016461
3: -0.0049828, -0.0030242, -0.0049508, -0.0028470, -0.0017586, 0.0015521
4: -0.0004245, 0.0013573, -0.0004908, 0.0013226, -0.0013106, 0.0013824
5: 0.0027190, 0.0044831, 0.0026918, 0.0045349, -0.0014197, 0.0014427
6: -0.0111788, -0.0045129, -0.0111019, -0.0043071, -0.0051458, 0.0049851
7: 0.0032011, 0.0125278, 0.0027882, 0.0123507, -0.0069353, 0.0073953
8: 0.9917424, 0.9980700, 0.9915449, 0.9979613, -0.0046480, 0.0048396
9: -0.0141070, -0.0082648, -0.0139937, -0.0080426, -0.0045445, 0.0043026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024935, upper bound: 0.0023592
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024935, upper bound: 0.0023739
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061467, 0.0094144, 0.0060821, 0.0094412, -0.0028039, 0.0027655
1: 0.0017325, 0.0026287, 0.0016433, 0.0026168, -0.0008179, 0.0009363
2: 0.0090599, 0.0109615, 0.0090197, 0.0109972, -0.0016647, 0.0017392
3: -0.0049996, -0.0028977, -0.0049525, -0.0027898, -0.0018603, 0.0016731
4: -0.0004607, 0.0013754, -0.0005070, 0.0013244, -0.0013282, 0.0014388
5: 0.0026696, 0.0045083, 0.0026704, 0.0045462, -0.0015148, 0.0014878
6: -0.0112674, -0.0044127, -0.0111249, -0.0042625, -0.0053736, 0.0050262
7: 0.0029564, 0.0126207, 0.0026785, 0.0123600, -0.0070953, 0.0077115
8: 0.9916462, 0.9981434, 0.9915022, 0.9979727, -0.0046784, 0.0050329
9: -0.0141664, -0.0081420, -0.0139997, -0.0079876, -0.0047309, 0.0043652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0023776
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0023921
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022225
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022549
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022773, upper bound: 0.0022360
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022773, upper bound: 0.0022684
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0023523
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0024261
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0023717
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0024458
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023092, upper bound: 0.0022225
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023092, upper bound: 0.0022337
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023627, upper bound: 0.0022361
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023627, upper bound: 0.0022485
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0023470
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0023601
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024458, upper bound: 0.0023663
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024458, upper bound: 0.0023793
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0022883
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022480, upper bound: 0.0023314
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0022959
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0023395
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0024548
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023329, upper bound: 0.0025268
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0024720
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023663, upper bound: 0.0025452
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023093, upper bound: 0.0022864
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023092, upper bound: 0.0022986
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023622, upper bound: 0.0022945
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023622, upper bound: 0.0023081
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0024510
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023889, upper bound: 0.0024631
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024698
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024813
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023493, upper bound: 0.0022225
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023493, upper bound: 0.0022549
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023832, upper bound: 0.0022361
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023832, upper bound: 0.0022684
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024263, upper bound: 0.0023523
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024263, upper bound: 0.0024250
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024698, upper bound: 0.0023717
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024698, upper bound: 0.0024449
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024105, upper bound: 0.0022225
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024105, upper bound: 0.0022336
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024622, upper bound: 0.0022361
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024622, upper bound: 0.0022484
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024866, upper bound: 0.0023470
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024866, upper bound: 0.0023601
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0025452, upper bound: 0.0023663
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0025452, upper bound: 0.0023793
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0022284
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0022604
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023833, upper bound: 0.0022353
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0023833, upper bound: 0.0022684
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024332, upper bound: 0.0023644
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024332, upper bound: 0.0024403
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024743, upper bound: 0.0023822
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024743, upper bound: 0.0024579
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024131, upper bound: 0.0022282
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024131, upper bound: 0.0022428
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024624, upper bound: 0.0022352
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024624, upper bound: 0.0022506
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024935, upper bound: 0.0023592
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0024935, upper bound: 0.0023739
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0023776
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0023921

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0064653, 0.0088359, 0.0065180, 0.0088502, -0.0018387, 0.0017811
1: 0.0022563, 0.0025988, 0.0022640, 0.0026009, -0.0002656, 0.0002573
2: 0.0094747, 0.0107854, 0.0094668, 0.0107563, -0.0009847, 0.0010166
3: -0.0048812, -0.0035257, -0.0048894, -0.0035558, -0.0010184, 0.0010514
4: -0.0002202, 0.0012473, -0.0001876, 0.0012561, -0.0011382, 0.0011025
5: 0.0029330, 0.0043217, 0.0029246, 0.0042908, -0.0010433, 0.0010771
6: -0.0106630, -0.0051531, -0.0106962, -0.0052755, -0.0041397, 0.0042737
7: 0.0044614, 0.0119653, 0.0046281, 0.0120106, -0.0058205, 0.0056379
8: 0.9923565, 0.9976425, 0.9924740, 0.9976744, -0.0041001, 0.0039714
9: -0.0137473, -0.0089491, -0.0137763, -0.0090557, -0.0036050, 0.0037218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0021081
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0021081
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0064653, 0.0088359, 0.0064867, 0.0089338, -0.0019791, 0.0018810
1: 0.0022563, 0.0025988, 0.0022594, 0.0026130, -0.0002859, 0.0002718
2: 0.0094747, 0.0107854, 0.0094206, 0.0107736, -0.0010400, 0.0010942
3: -0.0048812, -0.0035257, -0.0049373, -0.0035379, -0.0010756, 0.0011317
4: -0.0002202, 0.0012473, -0.0002069, 0.0013079, -0.0012251, 0.0011644
5: 0.0029330, 0.0043217, 0.0028756, 0.0043092, -0.0011019, 0.0011594
6: -0.0106630, -0.0051531, -0.0108907, -0.0052028, -0.0043720, 0.0046000
7: 0.0044614, 0.0119653, 0.0045290, 0.0122754, -0.0062648, 0.0059544
8: 0.9923565, 0.9976425, 0.9924042, 0.9978609, -0.0044131, 0.0041944
9: -0.0137473, -0.0089491, -0.0139456, -0.0089923, -0.0038074, 0.0040059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0021406
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0021406
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0064343, 0.0088552, 0.0065039, 0.0088541, -0.0018404, 0.0018546
1: 0.0022519, 0.0026016, 0.0022619, 0.0026015, -0.0002659, 0.0002679
2: 0.0094641, 0.0108025, 0.0094646, 0.0107640, -0.0010254, 0.0010175
3: -0.0048923, -0.0035080, -0.0048917, -0.0035478, -0.0010605, 0.0010524
4: -0.0002393, 0.0012592, -0.0001962, 0.0012586, -0.0011393, 0.0011481
5: 0.0029217, 0.0043398, 0.0029223, 0.0042991, -0.0010864, 0.0010781
6: -0.0107079, -0.0050811, -0.0107055, -0.0052429, -0.0043107, 0.0042777
7: 0.0043633, 0.0120266, 0.0045837, 0.0120232, -0.0058259, 0.0058708
8: 0.9922875, 0.9976856, 0.9924428, 0.9976832, -0.0041039, 0.0041355
9: -0.0137865, -0.0088864, -0.0137843, -0.0090273, -0.0037540, 0.0037252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0021211
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0021210
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0064343, 0.0088552, 0.0064688, 0.0089375, -0.0019797, 0.0019554
1: 0.0022519, 0.0026016, 0.0022569, 0.0026135, -0.0002860, 0.0002825
2: 0.0094641, 0.0108025, 0.0094186, 0.0107834, -0.0010811, 0.0010945
3: -0.0048923, -0.0035080, -0.0049394, -0.0035277, -0.0011181, 0.0011320
4: -0.0002393, 0.0012592, -0.0002180, 0.0013102, -0.0012255, 0.0012104
5: 0.0029217, 0.0043398, 0.0028735, 0.0043196, -0.0011455, 0.0011597
6: -0.0107079, -0.0050811, -0.0108992, -0.0051613, -0.0045449, 0.0046015
7: 0.0043633, 0.0120266, 0.0044726, 0.0122871, -0.0062668, 0.0061897
8: 0.9922875, 0.9976856, 0.9923645, 0.9978692, -0.0044145, 0.0043602
9: -0.0137865, -0.0088864, -0.0139530, -0.0089562, -0.0039579, 0.0040072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0021513
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0021513
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0064040, 0.0089179, 0.0063904, 0.0089232, -0.0020527, 0.0019303
1: 0.0021298, 0.0025988, 0.0021438, 0.0026021, -0.0004097, 0.0003697
2: 0.0094137, 0.0108192, 0.0094126, 0.0108268, -0.0010922, 0.0011517
3: -0.0048812, -0.0033818, -0.0048943, -0.0033862, -0.0011379, 0.0012422
4: -0.0002688, 0.0012473, -0.0002762, 0.0012614, -0.0012281, 0.0011292
5: 0.0028993, 0.0043576, 0.0028938, 0.0043656, -0.0011088, 0.0011875
6: -0.0106946, -0.0050108, -0.0107428, -0.0049791, -0.0042455, 0.0046028
7: 0.0041444, 0.0119654, 0.0041136, 0.0120376, -0.0063540, 0.0058404
8: 0.9922200, 0.9976521, 0.9921896, 0.9977021, -0.0043934, 0.0040450
9: -0.0137473, -0.0087849, -0.0137935, -0.0087613, -0.0036973, 0.0040211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0022712
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023202
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0064040, 0.0089179, 0.0063323, 0.0092552, -0.0025040, 0.0020287
1: 0.0021298, 0.0025988, 0.0018112, 0.0026171, -0.0004293, 0.0007353
2: 0.0094137, 0.0108192, 0.0091739, 0.0108589, -0.0011466, 0.0014854
3: -0.0048812, -0.0033818, -0.0049537, -0.0030479, -0.0015244, 0.0013199
4: -0.0002688, 0.0012473, -0.0003418, 0.0013257, -0.0013122, 0.0012216
5: 0.0028993, 0.0043576, 0.0027449, 0.0043996, -0.0011664, 0.0013874
6: -0.0106946, -0.0050108, -0.0110627, -0.0048439, -0.0044743, 0.0050217
7: 0.0041444, 0.0119654, 0.0035861, 0.0123662, -0.0067841, 0.0065143
8: 0.9922200, 0.9976521, 0.9920599, 0.9979573, -0.0047284, 0.0042644
9: -0.0137473, -0.0087849, -0.0140036, -0.0085323, -0.0040144, 0.0042961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023475
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023525
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063700, 0.0090073, 0.0063760, 0.0089633, -0.0021062, 0.0021121
1: 0.0020317, 0.0026016, 0.0020952, 0.0026026, -0.0005154, 0.0004330
2: 0.0093507, 0.0108381, 0.0093836, 0.0108347, -0.0012167, 0.0011940
3: -0.0048923, -0.0032743, -0.0048963, -0.0033340, -0.0012298, 0.0013425
4: -0.0002985, 0.0012593, -0.0002895, 0.0012636, -0.0012428, 0.0011819
5: 0.0028601, 0.0043775, 0.0028776, 0.0043740, -0.0011953, 0.0012101
6: -0.0107636, -0.0049316, -0.0107630, -0.0049457, -0.0044667, 0.0046433
7: 0.0039353, 0.0120267, 0.0040167, 0.0120488, -0.0064893, 0.0061405
8: 0.9921440, 0.9977032, 0.9921576, 0.9977137, -0.0044192, 0.0042305
9: -0.0137866, -0.0086834, -0.0138007, -0.0087156, -0.0038719, 0.0040735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0022856
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023388
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063700, 0.0090073, 0.0063117, 0.0093024, -0.0025878, 0.0022103
1: 0.0020317, 0.0026016, 0.0017582, 0.0026176, -0.0005344, 0.0008001
2: 0.0093507, 0.0108381, 0.0091368, 0.0108703, -0.0012710, 0.0015555
3: -0.0048923, -0.0032743, -0.0049556, -0.0029897, -0.0016225, 0.0014176
4: -0.0002985, 0.0012593, -0.0003581, 0.0013277, -0.0013241, 0.0012744
5: 0.0028601, 0.0043775, 0.0027261, 0.0044117, -0.0012529, 0.0014175
6: -0.0107636, -0.0049316, -0.0110830, -0.0047961, -0.0046950, 0.0050482
7: 0.0039353, 0.0120267, 0.0034766, 0.0123768, -0.0069050, 0.0068157
8: 0.9921440, 0.9977032, 0.9920141, 0.9979684, -0.0047416, 0.0044495
9: -0.0137866, -0.0086834, -0.0140104, -0.0084773, -0.0041892, 0.0043393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023627
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023712
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0064264, 0.0090096, 0.0065180, 0.0088502, -0.0019357, 0.0020344
1: 0.0021116, 0.0026122, 0.0022640, 0.0026009, -0.0004280, 0.0002761
2: 0.0093637, 0.0108069, 0.0094668, 0.0107563, -0.0011514, 0.0010702
3: -0.0049342, -0.0033694, -0.0048894, -0.0035558, -0.0010926, 0.0012482
4: -0.0002576, 0.0013046, -0.0001876, 0.0012561, -0.0012123, 0.0011828
5: 0.0028450, 0.0043445, 0.0029246, 0.0042908, -0.0011689, 0.0011339
6: -0.0109112, -0.0050628, -0.0106962, -0.0052755, -0.0044836, 0.0044991
7: 0.0041839, 0.0122585, 0.0046281, 0.0120106, -0.0062891, 0.0060484
8: 0.9922699, 0.9978596, 0.9924740, 0.9976744, -0.0043163, 0.0042745
9: -0.0139348, -0.0088202, -0.0137763, -0.0090557, -0.0038675, 0.0039705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021077
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021077
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0064264, 0.0090096, 0.0064867, 0.0089338, -0.0018783, 0.0019697
1: 0.0021116, 0.0026122, 0.0022594, 0.0026130, -0.0004158, 0.0002628
2: 0.0093637, 0.0108069, 0.0094206, 0.0107736, -0.0011197, 0.0010385
3: -0.0049342, -0.0033694, -0.0049373, -0.0035379, -0.0010403, 0.0012093
4: -0.0002576, 0.0013046, -0.0002069, 0.0013079, -0.0011760, 0.0011262
5: 0.0028450, 0.0043445, 0.0028756, 0.0043092, -0.0011256, 0.0011003
6: -0.0109112, -0.0050628, -0.0108907, -0.0052028, -0.0042839, 0.0043658
7: 0.0041839, 0.0122585, 0.0045290, 0.0122754, -0.0060993, 0.0057591
8: 0.9922699, 0.9978596, 0.9924042, 0.9978609, -0.0041884, 0.0040742
9: -0.0139348, -0.0088202, -0.0139456, -0.0089923, -0.0036825, 0.0038517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021128
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021126
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063869, 0.0091103, 0.0065039, 0.0088541, -0.0019351, 0.0022297
1: 0.0019941, 0.0026166, 0.0022619, 0.0026015, -0.0005493, 0.0002875
2: 0.0092884, 0.0108287, 0.0094646, 0.0107640, -0.0012827, 0.0010698
3: -0.0049515, -0.0032459, -0.0048917, -0.0035478, -0.0011379, 0.0013626
4: -0.0002918, 0.0013233, -0.0001962, 0.0012586, -0.0012229, 0.0012318
5: 0.0028010, 0.0043676, 0.0029223, 0.0042991, -0.0012607, 0.0011336
6: -0.0110097, -0.0049708, -0.0107055, -0.0052429, -0.0047080, 0.0044976
7: 0.0039447, 0.0123542, 0.0045837, 0.0120232, -0.0064128, 0.0062991
8: 0.9921817, 0.9979361, 0.9924428, 0.9976832, -0.0043148, 0.0044636
9: -0.0139960, -0.0087035, -0.0137843, -0.0090273, -0.0040278, 0.0040105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022447, upper bound: 0.0021210
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022464, upper bound: 0.0021208
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063869, 0.0091103, 0.0064688, 0.0089375, -0.0018783, 0.0021721
1: 0.0019941, 0.0026166, 0.0022569, 0.0026135, -0.0005368, 0.0002736
2: 0.0092884, 0.0108287, 0.0094186, 0.0107834, -0.0012600, 0.0010384
3: -0.0049515, -0.0032459, -0.0049394, -0.0035277, -0.0010829, 0.0013190
4: -0.0002918, 0.0013233, -0.0002180, 0.0013102, -0.0011869, 0.0011723
5: 0.0028010, 0.0043676, 0.0028735, 0.0043196, -0.0012230, 0.0011003
6: -0.0110097, -0.0049708, -0.0108992, -0.0051613, -0.0044968, 0.0043656
7: 0.0039447, 0.0123542, 0.0044726, 0.0122871, -0.0062225, 0.0059947
8: 0.9921817, 0.9979361, 0.9923645, 0.9978692, -0.0041882, 0.0042512
9: -0.0139960, -0.0087035, -0.0139530, -0.0089562, -0.0038331, 0.0038923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022447, upper bound: 0.0021265
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022464, upper bound: 0.0021259
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063535, 0.0092428, 0.0063904, 0.0089232, -0.0021449, 0.0024314
1: 0.0018053, 0.0026122, 0.0021438, 0.0026021, -0.0007664, 0.0003895
2: 0.0091731, 0.0108472, 0.0094126, 0.0108268, -0.0014642, 0.0012027
3: -0.0049342, -0.0030505, -0.0048943, -0.0033862, -0.0012163, 0.0016185
4: -0.0003296, 0.0013046, -0.0002762, 0.0012614, -0.0013163, 0.0012141
5: 0.0027548, 0.0043872, 0.0028938, 0.0043656, -0.0013299, 0.0012415
6: -0.0109878, -0.0048933, -0.0107428, -0.0049791, -0.0046860, 0.0048173
7: 0.0036435, 0.0122588, 0.0041136, 0.0120376, -0.0070049, 0.0062743
8: 0.9921073, 0.9978809, 0.9921896, 0.9977021, -0.0045992, 0.0043850
9: -0.0139349, -0.0085719, -0.0137935, -0.0087613, -0.0039747, 0.0043246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0022637
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023202
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063535, 0.0092428, 0.0063323, 0.0092552, -0.0024878, 0.0024412
1: 0.0018053, 0.0026122, 0.0018112, 0.0026171, -0.0007524, 0.0007224
2: 0.0091731, 0.0108472, 0.0091739, 0.0108589, -0.0014888, 0.0014895
3: -0.0049342, -0.0030505, -0.0049537, -0.0030479, -0.0014831, 0.0015771
4: -0.0003296, 0.0013046, -0.0003418, 0.0013257, -0.0012828, 0.0011893
5: 0.0027548, 0.0043872, 0.0027449, 0.0043996, -0.0013144, 0.0013600
6: -0.0109878, -0.0048933, -0.0110627, -0.0048439, -0.0045054, 0.0048290
7: 0.0036435, 0.0122588, 0.0035861, 0.0123662, -0.0068254, 0.0063403
8: 0.9921073, 0.9978809, 0.9920599, 0.9979573, -0.0045230, 0.0041884
9: -0.0139349, -0.0085719, -0.0140036, -0.0085323, -0.0039080, 0.0042143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0022782
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023317
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063077, 0.0093598, 0.0063760, 0.0089633, -0.0021962, 0.0026431
1: 0.0016922, 0.0026166, 0.0020952, 0.0026026, -0.0008823, 0.0004532
2: 0.0090791, 0.0108725, 0.0093836, 0.0108347, -0.0016198, 0.0012437
3: -0.0049515, -0.0029236, -0.0048963, -0.0033340, -0.0013098, 0.0017378
4: -0.0003664, 0.0013234, -0.0002895, 0.0012636, -0.0013312, 0.0012686
5: 0.0027086, 0.0044140, 0.0028776, 0.0043740, -0.0014222, 0.0012628
6: -0.0110823, -0.0047868, -0.0107630, -0.0049457, -0.0049075, 0.0048526
7: 0.0033997, 0.0123546, 0.0040167, 0.0120488, -0.0071505, 0.0065834
8: 0.9920052, 0.9979572, 0.9921576, 0.9977137, -0.0046199, 0.0045720
9: -0.0139962, -0.0084475, -0.0138007, -0.0087156, -0.0041551, 0.0043782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0022773
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023388
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063077, 0.0093598, 0.0063117, 0.0093024, -0.0025862, 0.0026688
1: 0.0016922, 0.0026166, 0.0017582, 0.0026176, -0.0008692, 0.0007879
2: 0.0090791, 0.0108725, 0.0091368, 0.0108703, -0.0016611, 0.0015705
3: -0.0049515, -0.0029236, -0.0049556, -0.0029897, -0.0015839, 0.0016963
4: -0.0003664, 0.0013234, -0.0003581, 0.0013277, -0.0012992, 0.0012420
5: 0.0027086, 0.0044140, 0.0027261, 0.0044117, -0.0014120, 0.0014004
6: -0.0110823, -0.0047868, -0.0110830, -0.0047961, -0.0047269, 0.0048666
7: 0.0033997, 0.0123546, 0.0034766, 0.0123768, -0.0069782, 0.0066461
8: 0.9920052, 0.9979572, 0.9920141, 0.9979684, -0.0045482, 0.0043765
9: -0.0139962, -0.0084475, -0.0140104, -0.0084773, -0.0040828, 0.0042729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0022925
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023504
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0064653, 0.0088359, 0.0063602, 0.0089282, -0.0019335, 0.0019676
1: 0.0022563, 0.0025988, 0.0022412, 0.0026122, -0.0002793, 0.0002843
2: 0.0094747, 0.0107854, 0.0094237, 0.0108435, -0.0010878, 0.0010690
3: -0.0048812, -0.0035257, -0.0049340, -0.0034656, -0.0011251, 0.0011056
4: -0.0002202, 0.0012473, -0.0002852, 0.0013044, -0.0011969, 0.0012180
5: 0.0029330, 0.0043217, 0.0028789, 0.0043833, -0.0011526, 0.0011327
6: -0.0106630, -0.0051531, -0.0108775, -0.0049088, -0.0045732, 0.0044941
7: 0.0044614, 0.0119653, 0.0041287, 0.0122575, -0.0061206, 0.0062283
8: 0.9923565, 0.9976425, 0.9921222, 0.9978483, -0.0043114, 0.0043873
9: -0.0137473, -0.0089491, -0.0139341, -0.0087363, -0.0039825, 0.0039137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0021759
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0021758
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0064653, 0.0088359, 0.0063354, 0.0090067, -0.0020738, 0.0020542
1: 0.0022563, 0.0025988, 0.0022376, 0.0026235, -0.0002996, 0.0002968
2: 0.0094747, 0.0107854, 0.0093803, 0.0108572, -0.0011357, 0.0011466
3: -0.0048812, -0.0035257, -0.0049789, -0.0034514, -0.0011746, 0.0011858
4: -0.0002202, 0.0012473, -0.0003006, 0.0013530, -0.0012837, 0.0012716
5: 0.0029330, 0.0043217, 0.0028329, 0.0043978, -0.0012034, 0.0012149
6: -0.0106630, -0.0051531, -0.0110601, -0.0048512, -0.0047746, 0.0048202
7: 0.0044614, 0.0119653, 0.0040501, 0.0125061, -0.0065647, 0.0065026
8: 0.9923565, 0.9976425, 0.9920669, 0.9980235, -0.0046243, 0.0045806
9: -0.0137473, -0.0089491, -0.0140931, -0.0086861, -0.0041579, 0.0041976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0022167
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0022163
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0064343, 0.0088552, 0.0063486, 0.0089314, -0.0019437, 0.0020354
1: 0.0022519, 0.0026016, 0.0022395, 0.0026126, -0.0002808, 0.0002941
2: 0.0094641, 0.0108025, 0.0094219, 0.0108499, -0.0011253, 0.0010746
3: -0.0048923, -0.0035080, -0.0049359, -0.0034590, -0.0011639, 0.0011114
4: -0.0002393, 0.0012592, -0.0002924, 0.0013064, -0.0012032, 0.0012599
5: 0.0029217, 0.0043398, 0.0028770, 0.0043901, -0.0011923, 0.0011386
6: -0.0107079, -0.0050811, -0.0108851, -0.0048818, -0.0047308, 0.0045176
7: 0.0043633, 0.0120266, 0.0040919, 0.0122678, -0.0061526, 0.0064430
8: 0.9922875, 0.9976856, 0.9920964, 0.9978556, -0.0043340, 0.0045386
9: -0.0137865, -0.0088864, -0.0139407, -0.0087128, -0.0041198, 0.0039341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0021896
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0021896
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0064343, 0.0088552, 0.0063192, 0.0090098, -0.0020779, 0.0021287
1: 0.0022519, 0.0026016, 0.0022352, 0.0026240, -0.0003002, 0.0003075
2: 0.0094641, 0.0108025, 0.0093786, 0.0108661, -0.0011769, 0.0011488
3: -0.0048923, -0.0035080, -0.0049807, -0.0034422, -0.0012172, 0.0011882
4: -0.0002393, 0.0012592, -0.0003106, 0.0013549, -0.0012863, 0.0013177
5: 0.0029217, 0.0043398, 0.0028311, 0.0044072, -0.0012470, 0.0012172
6: -0.0107079, -0.0050811, -0.0110673, -0.0048137, -0.0049476, 0.0048297
7: 0.0043633, 0.0120266, 0.0039991, 0.0125159, -0.0065776, 0.0067382
8: 0.9922875, 0.9976856, 0.9920309, 0.9980304, -0.0046334, 0.0047465
9: -0.0137865, -0.0088864, -0.0140994, -0.0086535, -0.0043086, 0.0042059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0022299
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0022299
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0064040, 0.0089179, 0.0062168, 0.0089984, -0.0021143, 0.0021192
1: 0.0021298, 0.0025988, 0.0021610, 0.0026162, -0.0004239, 0.0003526
2: 0.0094137, 0.0108192, 0.0093775, 0.0109228, -0.0011966, 0.0011784
3: -0.0048812, -0.0033818, -0.0049498, -0.0033262, -0.0012025, 0.0012992
4: -0.0002688, 0.0012473, -0.0003797, 0.0013215, -0.0012895, 0.0012421
5: 0.0028993, 0.0043576, 0.0028449, 0.0044673, -0.0012194, 0.0012303
6: -0.0106946, -0.0050108, -0.0109608, -0.0045755, -0.0046846, 0.0048212
7: 0.0041444, 0.0119654, 0.0036087, 0.0123451, -0.0066689, 0.0063905
8: 0.9922200, 0.9976521, 0.9918025, 0.9979163, -0.0046106, 0.0044662
9: -0.0137473, -0.0087849, -0.0139902, -0.0084245, -0.0040644, 0.0042221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023804
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023827
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0064040, 0.0089179, 0.0061675, 0.0093072, -0.0025524, 0.0022119
1: 0.0021298, 0.0025988, 0.0018532, 0.0026293, -0.0004434, 0.0006904
2: 0.0094137, 0.0108192, 0.0091526, 0.0109500, -0.0012479, 0.0014968
3: -0.0048812, -0.0033818, -0.0050020, -0.0030221, -0.0015623, 0.0013766
4: -0.0002688, 0.0012473, -0.0004372, 0.0013780, -0.0013733, 0.0013296
5: 0.0028993, 0.0043576, 0.0027075, 0.0044961, -0.0012737, 0.0014204
6: -0.0106946, -0.0050108, -0.0112478, -0.0044610, -0.0049000, 0.0052221
7: 0.0041444, 0.0119654, 0.0031424, 0.0126341, -0.0070973, 0.0070327
8: 0.9922200, 0.9976521, 0.9916925, 0.9981436, -0.0049391, 0.0046728
9: -0.0137473, -0.0087849, -0.0141749, -0.0082238, -0.0043651, 0.0044960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0024529
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0024258
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063700, 0.0090073, 0.0061998, 0.0090350, -0.0021755, 0.0022978
1: 0.0020317, 0.0026016, 0.0021123, 0.0026166, -0.0005298, 0.0004145
2: 0.0093507, 0.0108381, 0.0093505, 0.0109322, -0.0013193, 0.0012221
3: -0.0048923, -0.0032743, -0.0049515, -0.0032740, -0.0012951, 0.0014001
4: -0.0002985, 0.0012593, -0.0003946, 0.0013233, -0.0013048, 0.0012928
5: 0.0028601, 0.0043775, 0.0028291, 0.0044772, -0.0013041, 0.0012574
6: -0.0107636, -0.0049316, -0.0109819, -0.0045360, -0.0048983, 0.0048652
7: 0.0039353, 0.0120267, 0.0035047, 0.0123543, -0.0068056, 0.0066796
8: 0.9921440, 0.9977032, 0.9917645, 0.9979273, -0.0046400, 0.0046445
9: -0.0137866, -0.0086834, -0.0139960, -0.0083737, -0.0042324, 0.0042761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023893
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023996
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063700, 0.0090073, 0.0061482, 0.0093571, -0.0026373, 0.0023946
1: 0.0020317, 0.0026016, 0.0018013, 0.0026298, -0.0005490, 0.0007549
2: 0.0093507, 0.0108381, 0.0091161, 0.0109607, -0.0013729, 0.0015623
3: -0.0048923, -0.0032743, -0.0050038, -0.0029650, -0.0016560, 0.0014763
4: -0.0002985, 0.0012593, -0.0004536, 0.0013799, -0.0013873, 0.0013832
5: 0.0028601, 0.0043775, 0.0026892, 0.0045074, -0.0013608, 0.0014544
6: -0.0107636, -0.0049316, -0.0112672, -0.0044162, -0.0051233, 0.0052648
7: 0.0039353, 0.0120267, 0.0030311, 0.0126436, -0.0072277, 0.0073371
8: 0.9921440, 0.9977032, 0.9916496, 0.9981542, -0.0049641, 0.0048604
9: -0.0137866, -0.0086834, -0.0141810, -0.0081681, -0.0045426, 0.0045460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0024622
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0024439
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0064264, 0.0090096, 0.0063602, 0.0089282, -0.0020327, 0.0022209
1: 0.0021116, 0.0026122, 0.0022412, 0.0026122, -0.0004419, 0.0003030
2: 0.0093637, 0.0108069, 0.0094237, 0.0108435, -0.0012546, 0.0011238
3: -0.0049342, -0.0033694, -0.0049340, -0.0034656, -0.0011992, 0.0013033
4: -0.0002576, 0.0013046, -0.0002852, 0.0013044, -0.0012723, 0.0012982
5: 0.0028450, 0.0043445, 0.0028789, 0.0043833, -0.0012781, 0.0011908
6: -0.0109112, -0.0050628, -0.0108775, -0.0049088, -0.0049171, 0.0047246
7: 0.0041839, 0.0122585, 0.0041287, 0.0122575, -0.0065953, 0.0066388
8: 0.9922699, 0.9978596, 0.9921222, 0.9978483, -0.0045326, 0.0046904
9: -0.0139348, -0.0088202, -0.0139341, -0.0087363, -0.0042451, 0.0041669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021750
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021750
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0064264, 0.0090096, 0.0063354, 0.0090067, -0.0019749, 0.0021520
1: 0.0021116, 0.0026122, 0.0022376, 0.0026235, -0.0004296, 0.0002892
2: 0.0093637, 0.0108069, 0.0093803, 0.0108572, -0.0012205, 0.0010919
3: -0.0049342, -0.0033694, -0.0049789, -0.0034514, -0.0011446, 0.0012650
4: -0.0002576, 0.0013046, -0.0003006, 0.0013530, -0.0012357, 0.0012391
5: 0.0028450, 0.0043445, 0.0028329, 0.0043978, -0.0012325, 0.0011569
6: -0.0109112, -0.0050628, -0.0110601, -0.0048512, -0.0047077, 0.0045901
7: 0.0041839, 0.0122585, 0.0040501, 0.0125061, -0.0064044, 0.0063363
8: 0.9922699, 0.9978596, 0.9920669, 0.9980235, -0.0044036, 0.0044808
9: -0.0139348, -0.0088202, -0.0140931, -0.0086861, -0.0040516, 0.0040469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021796
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021794
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063869, 0.0091103, 0.0063486, 0.0089314, -0.0020398, 0.0024104
1: 0.0019941, 0.0026166, 0.0022395, 0.0026126, -0.0005640, 0.0003136
2: 0.0092884, 0.0108287, 0.0094219, 0.0108499, -0.0013826, 0.0011277
3: -0.0049515, -0.0032459, -0.0049359, -0.0034590, -0.0012412, 0.0014217
4: -0.0002918, 0.0013233, -0.0002924, 0.0013064, -0.0012877, 0.0013437
5: 0.0028010, 0.0043676, 0.0028770, 0.0043901, -0.0013666, 0.0011949
6: -0.0110097, -0.0049708, -0.0108851, -0.0048818, -0.0051281, 0.0047410
7: 0.0039447, 0.0123542, 0.0040919, 0.0122678, -0.0067444, 0.0068713
8: 0.9921817, 0.9979361, 0.9920964, 0.9978556, -0.0045483, 0.0048666
9: -0.0139960, -0.0087035, -0.0139407, -0.0087128, -0.0043937, 0.0042225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022445, upper bound: 0.0021889
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022456, upper bound: 0.0021888
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063869, 0.0091103, 0.0063192, 0.0090098, -0.0019824, 0.0023500
1: 0.0019941, 0.0026166, 0.0022352, 0.0026240, -0.0005518, 0.0002993
2: 0.0092884, 0.0108287, 0.0093786, 0.0108661, -0.0013584, 0.0010960
3: -0.0049515, -0.0032459, -0.0049807, -0.0034422, -0.0011846, 0.0013786
4: -0.0002918, 0.0013233, -0.0003106, 0.0013549, -0.0012512, 0.0012824
5: 0.0028010, 0.0043676, 0.0028311, 0.0044072, -0.0013272, 0.0011613
6: -0.0110097, -0.0049708, -0.0110673, -0.0048137, -0.0049104, 0.0046076
7: 0.0039447, 0.0123542, 0.0039991, 0.0125159, -0.0065511, 0.0065579
8: 0.9921817, 0.9979361, 0.9920309, 0.9980304, -0.0044204, 0.0046480
9: -0.0139960, -0.0087035, -0.0140994, -0.0086535, -0.0041933, 0.0041027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022445, upper bound: 0.0021945
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022456, upper bound: 0.0021945
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063535, 0.0092428, 0.0062168, 0.0089984, -0.0022078, 0.0026203
1: 0.0018053, 0.0026122, 0.0021610, 0.0026162, -0.0007806, 0.0003724
2: 0.0091731, 0.0108472, 0.0093775, 0.0109228, -0.0015687, 0.0012301
3: -0.0049342, -0.0030505, -0.0049498, -0.0033262, -0.0012809, 0.0016753
4: -0.0003296, 0.0013046, -0.0003797, 0.0013215, -0.0013783, 0.0013270
5: 0.0027548, 0.0043872, 0.0028449, 0.0044673, -0.0014405, 0.0012851
6: -0.0109878, -0.0048933, -0.0109608, -0.0045755, -0.0051250, 0.0050385
7: 0.0036435, 0.0122588, 0.0036087, 0.0123451, -0.0073215, 0.0068244
8: 0.9921073, 0.9978809, 0.9918025, 0.9979163, -0.0048191, 0.0048063
9: -0.0139349, -0.0085719, -0.0139902, -0.0084245, -0.0043419, 0.0045271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023738
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023815
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063535, 0.0092428, 0.0061675, 0.0093072, -0.0025243, 0.0026287
1: 0.0018053, 0.0026122, 0.0018532, 0.0026293, -0.0007670, 0.0006782
2: 0.0091731, 0.0108472, 0.0091526, 0.0109500, -0.0015925, 0.0014869
3: -0.0049342, -0.0030505, -0.0050020, -0.0030221, -0.0015257, 0.0016340
4: -0.0003296, 0.0013046, -0.0004372, 0.0013780, -0.0013449, 0.0012976
5: 0.0027548, 0.0043872, 0.0027075, 0.0044961, -0.0014242, 0.0013966
6: -0.0109878, -0.0048933, -0.0112478, -0.0044610, -0.0049411, 0.0050307
7: 0.0036435, 0.0122588, 0.0031424, 0.0126341, -0.0071431, 0.0068520
8: 0.9921073, 0.9978809, 0.9916925, 0.9981436, -0.0047409, 0.0046065
9: -0.0139349, -0.0085719, -0.0141749, -0.0082238, -0.0042588, 0.0044174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023876
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023921
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063077, 0.0093598, 0.0061998, 0.0090350, -0.0022665, 0.0028288
1: 0.0016922, 0.0026166, 0.0021123, 0.0026166, -0.0008966, 0.0004348
2: 0.0090791, 0.0108725, 0.0093505, 0.0109322, -0.0017225, 0.0012724
3: -0.0049515, -0.0029236, -0.0049515, -0.0032740, -0.0013751, 0.0017944
4: -0.0003664, 0.0013234, -0.0003946, 0.0013233, -0.0013935, 0.0013794
5: 0.0027086, 0.0044140, 0.0028291, 0.0044772, -0.0015310, 0.0013107
6: -0.0110823, -0.0047868, -0.0109819, -0.0045360, -0.0053390, 0.0050766
7: 0.0033997, 0.0123546, 0.0035047, 0.0123543, -0.0074692, 0.0071225
8: 0.9920052, 0.9979572, 0.9917645, 0.9979273, -0.0048428, 0.0049860
9: -0.0139962, -0.0084475, -0.0139960, -0.0083737, -0.0045156, 0.0045821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023832
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023991
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063077, 0.0093598, 0.0061482, 0.0093571, -0.0026173, 0.0028545
1: 0.0016922, 0.0026166, 0.0018013, 0.0026298, -0.0008838, 0.0007430
2: 0.0090791, 0.0108725, 0.0091161, 0.0109607, -0.0017638, 0.0015606
3: -0.0049515, -0.0029236, -0.0050038, -0.0029650, -0.0016211, 0.0017531
4: -0.0003664, 0.0013234, -0.0004536, 0.0013799, -0.0013613, 0.0013507
5: 0.0027086, 0.0044140, 0.0026892, 0.0045074, -0.0015208, 0.0014345
6: -0.0110823, -0.0047868, -0.0112672, -0.0044162, -0.0051587, 0.0050888
7: 0.0033997, 0.0123546, 0.0030311, 0.0126436, -0.0072967, 0.0071613
8: 0.9920052, 0.9979572, 0.9916496, 0.9981542, -0.0047695, 0.0047907
9: -0.0139962, -0.0084475, -0.0141810, -0.0081681, -0.0044354, 0.0044760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023993
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0024089
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063038, 0.0089297, 0.0065180, 0.0088502, -0.0020294, 0.0018842
1: 0.0022330, 0.0026124, 0.0022640, 0.0026009, -0.0002932, 0.0002722
2: 0.0094229, 0.0108747, 0.0094668, 0.0107563, -0.0010417, 0.0011220
3: -0.0049349, -0.0034334, -0.0048894, -0.0035558, -0.0010774, 0.0011604
4: -0.0003202, 0.0013053, -0.0001876, 0.0012561, -0.0012562, 0.0011663
5: 0.0028781, 0.0044163, 0.0029246, 0.0042908, -0.0011038, 0.0011888
6: -0.0108810, -0.0047777, -0.0106962, -0.0052755, -0.0043794, 0.0047168
7: 0.0039501, 0.0122622, 0.0046281, 0.0120106, -0.0064239, 0.0059643
8: 0.9919963, 0.9978516, 0.9924740, 0.9976744, -0.0045251, 0.0042014
9: -0.0139372, -0.0086221, -0.0137763, -0.0090557, -0.0038137, 0.0041076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022423, upper bound: 0.0021081
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021081
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063038, 0.0089297, 0.0064867, 0.0089338, -0.0021698, 0.0019846
1: 0.0022330, 0.0026124, 0.0022594, 0.0026130, -0.0003135, 0.0002867
2: 0.0094229, 0.0108747, 0.0094206, 0.0107736, -0.0010972, 0.0011996
3: -0.0049349, -0.0034334, -0.0049373, -0.0035379, -0.0011348, 0.0012407
4: -0.0003202, 0.0013053, -0.0002069, 0.0013079, -0.0013431, 0.0012285
5: 0.0028781, 0.0044163, 0.0028756, 0.0043092, -0.0011626, 0.0012710
6: -0.0108810, -0.0047777, -0.0108907, -0.0052028, -0.0046128, 0.0050431
7: 0.0039501, 0.0122622, 0.0045290, 0.0122754, -0.0068683, 0.0062823
8: 0.9919963, 0.9978516, 0.9924042, 0.9978609, -0.0048382, 0.0044254
9: -0.0139372, -0.0086221, -0.0139456, -0.0089923, -0.0040170, 0.0043918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022423, upper bound: 0.0021406
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021406
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062742, 0.0089521, 0.0065039, 0.0088541, -0.0020252, 0.0019557
1: 0.0022287, 0.0026156, 0.0022619, 0.0026015, -0.0002926, 0.0002825
2: 0.0094105, 0.0108910, 0.0094646, 0.0107640, -0.0010812, 0.0011197
3: -0.0049477, -0.0034164, -0.0048917, -0.0035478, -0.0011183, 0.0011580
4: -0.0003385, 0.0013192, -0.0001962, 0.0012586, -0.0012536, 0.0012106
5: 0.0028649, 0.0044337, 0.0029223, 0.0042991, -0.0011456, 0.0011864
6: -0.0109331, -0.0047089, -0.0107055, -0.0052429, -0.0045456, 0.0047071
7: 0.0038564, 0.0123333, 0.0045837, 0.0120232, -0.0064107, 0.0061907
8: 0.9919304, 0.9979017, 0.9924428, 0.9976832, -0.0045158, 0.0043608
9: -0.0139826, -0.0085622, -0.0137843, -0.0090273, -0.0039585, 0.0040992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022747, upper bound: 0.0021211
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021210
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062742, 0.0089521, 0.0064688, 0.0089375, -0.0021645, 0.0020570
1: 0.0022287, 0.0026156, 0.0022569, 0.0026135, -0.0003127, 0.0002972
2: 0.0094105, 0.0108910, 0.0094186, 0.0107834, -0.0011372, 0.0011967
3: -0.0049477, -0.0034164, -0.0049394, -0.0035277, -0.0011762, 0.0012377
4: -0.0003385, 0.0013192, -0.0002180, 0.0013102, -0.0013399, 0.0012733
5: 0.0028649, 0.0044337, 0.0028735, 0.0043196, -0.0012050, 0.0012680
6: -0.0109331, -0.0047089, -0.0108992, -0.0051613, -0.0047810, 0.0050309
7: 0.0038564, 0.0123333, 0.0044726, 0.0122871, -0.0068517, 0.0065113
8: 0.9919304, 0.9979017, 0.9923645, 0.9978692, -0.0048265, 0.0045867
9: -0.0139826, -0.0085622, -0.0139530, -0.0089562, -0.0041635, 0.0043811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022747, upper bound: 0.0021513
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021513
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062336, 0.0089878, 0.0063904, 0.0089232, -0.0022458, 0.0020040
1: 0.0021440, 0.0026124, 0.0021438, 0.0026021, -0.0003963, 0.0003845
2: 0.0093805, 0.0109134, 0.0094126, 0.0108268, -0.0011250, 0.0012585
3: -0.0049349, -0.0033184, -0.0048943, -0.0033862, -0.0011971, 0.0013148
4: -0.0003711, 0.0013053, -0.0002762, 0.0012614, -0.0013439, 0.0011931
5: 0.0028538, 0.0044574, 0.0028938, 0.0043656, -0.0011582, 0.0013006
6: -0.0109073, -0.0046147, -0.0107428, -0.0049791, -0.0044788, 0.0050518
7: 0.0036416, 0.0122623, 0.0041136, 0.0120376, -0.0069231, 0.0061672
8: 0.9918400, 0.9978604, 0.9921896, 0.9977021, -0.0048241, 0.0042729
9: -0.0139372, -0.0084520, -0.0137935, -0.0087613, -0.0039061, 0.0043982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0022704
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0023202
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062336, 0.0089878, 0.0063323, 0.0092552, -0.0026971, 0.0021047
1: 0.0021440, 0.0026124, 0.0018112, 0.0026171, -0.0004160, 0.0007500
2: 0.0093805, 0.0109134, 0.0091739, 0.0108589, -0.0011806, 0.0015922
3: -0.0049349, -0.0033184, -0.0049537, -0.0030479, -0.0015835, 0.0013925
4: -0.0003711, 0.0013053, -0.0003418, 0.0013257, -0.0014280, 0.0012869
5: 0.0028538, 0.0044574, 0.0027449, 0.0043996, -0.0012172, 0.0015006
6: -0.0109073, -0.0046147, -0.0110627, -0.0048439, -0.0047130, 0.0054706
7: 0.0036416, 0.0122623, 0.0035861, 0.0123662, -0.0073532, 0.0068459
8: 0.9918400, 0.9978604, 0.9920599, 0.9979573, -0.0051591, 0.0044975
9: -0.0139372, -0.0084520, -0.0140036, -0.0085323, -0.0042276, 0.0046732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0023468
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0023525
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061939, 0.0090750, 0.0063760, 0.0089633, -0.0022960, 0.0021800
1: 0.0020511, 0.0026156, 0.0020952, 0.0026026, -0.0004938, 0.0004461
2: 0.0093184, 0.0109354, 0.0093836, 0.0108347, -0.0012465, 0.0012989
3: -0.0049477, -0.0032169, -0.0048963, -0.0033340, -0.0012823, 0.0014088
4: -0.0004035, 0.0013193, -0.0002895, 0.0012636, -0.0013562, 0.0012389
5: 0.0028141, 0.0044807, 0.0028776, 0.0043740, -0.0012395, 0.0013213
6: -0.0109852, -0.0045223, -0.0107630, -0.0049457, -0.0046742, 0.0050845
7: 0.0034260, 0.0123335, 0.0040167, 0.0120488, -0.0070412, 0.0064314
8: 0.9917513, 0.9979182, 0.9921576, 0.9977137, -0.0048424, 0.0044327
9: -0.0139827, -0.0083422, -0.0138007, -0.0087156, -0.0040582, 0.0044423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0022848
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0023388
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061939, 0.0090750, 0.0063117, 0.0093024, -0.0027777, 0.0022803
1: 0.0020511, 0.0026156, 0.0017582, 0.0026176, -0.0005128, 0.0008132
2: 0.0093184, 0.0109354, 0.0091368, 0.0108703, -0.0013020, 0.0016604
3: -0.0049477, -0.0032169, -0.0049556, -0.0029897, -0.0016749, 0.0014838
4: -0.0004035, 0.0013193, -0.0003581, 0.0013277, -0.0014374, 0.0013325
5: 0.0028141, 0.0044807, 0.0027261, 0.0044117, -0.0012983, 0.0015287
6: -0.0109852, -0.0045223, -0.0110830, -0.0047961, -0.0049075, 0.0054894
7: 0.0034260, 0.0123335, 0.0034766, 0.0123768, -0.0074569, 0.0071109
8: 0.9917513, 0.9979182, 0.9920141, 0.9979684, -0.0051649, 0.0046565
9: -0.0139827, -0.0083422, -0.0140104, -0.0084773, -0.0043792, 0.0047081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0023623
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0023712
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062713, 0.0090630, 0.0065180, 0.0088502, -0.0021148, 0.0020821
1: 0.0021509, 0.0026245, 0.0022640, 0.0026009, -0.0003897, 0.0002909
2: 0.0093401, 0.0108927, 0.0094668, 0.0107563, -0.0011642, 0.0011692
3: -0.0049828, -0.0033385, -0.0048894, -0.0035558, -0.0011512, 0.0012922
4: -0.0003480, 0.0013572, -0.0001876, 0.0012561, -0.0013174, 0.0012463
5: 0.0028084, 0.0044354, 0.0029246, 0.0042908, -0.0012074, 0.0012388
6: -0.0110978, -0.0047021, -0.0106962, -0.0052755, -0.0047058, 0.0049153
7: 0.0037583, 0.0125275, 0.0046281, 0.0120106, -0.0067894, 0.0063731
8: 0.9919239, 0.9980459, 0.9924740, 0.9976744, -0.0047156, 0.0044980
9: -0.0141068, -0.0085273, -0.0137763, -0.0090557, -0.0040752, 0.0043116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022983, upper bound: 0.0021077
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022981, upper bound: 0.0021077
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062713, 0.0090630, 0.0064867, 0.0089338, -0.0020665, 0.0020082
1: 0.0021509, 0.0026245, 0.0022594, 0.0026130, -0.0003789, 0.0002779
2: 0.0093401, 0.0108927, 0.0094206, 0.0107736, -0.0011261, 0.0011425
3: -0.0049828, -0.0033385, -0.0049373, -0.0035379, -0.0010999, 0.0012561
4: -0.0003480, 0.0013572, -0.0002069, 0.0013079, -0.0012867, 0.0011908
5: 0.0028084, 0.0044354, 0.0028756, 0.0043092, -0.0011614, 0.0012106
6: -0.0110978, -0.0047021, -0.0108907, -0.0052028, -0.0045051, 0.0048031
7: 0.0037583, 0.0125275, 0.0045290, 0.0122754, -0.0066270, 0.0060892
8: 0.9919239, 0.9980459, 0.9924042, 0.9978609, -0.0046079, 0.0043005
9: -0.0141068, -0.0085273, -0.0139456, -0.0089923, -0.0038936, 0.0042107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022983, upper bound: 0.0021128
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022981, upper bound: 0.0021126
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062344, 0.0091598, 0.0065039, 0.0088541, -0.0021144, 0.0022556
1: 0.0020382, 0.0026287, 0.0022619, 0.0026015, -0.0005035, 0.0003019
2: 0.0092712, 0.0109130, 0.0094646, 0.0107640, -0.0012854, 0.0011690
3: -0.0049995, -0.0032212, -0.0048917, -0.0035478, -0.0011951, 0.0013996
4: -0.0003801, 0.0013753, -0.0001962, 0.0012586, -0.0013278, 0.0012937
5: 0.0027642, 0.0044569, 0.0029223, 0.0042991, -0.0012896, 0.0012386
6: -0.0111929, -0.0046165, -0.0107055, -0.0052429, -0.0049159, 0.0049144
7: 0.0035344, 0.0126202, 0.0045837, 0.0120232, -0.0069115, 0.0066158
8: 0.9918417, 0.9981200, 0.9924428, 0.9976832, -0.0047147, 0.0046784
9: -0.0141661, -0.0084179, -0.0137843, -0.0090273, -0.0042303, 0.0043508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023423, upper bound: 0.0021210
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023522, upper bound: 0.0021208
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062344, 0.0091598, 0.0064688, 0.0089375, -0.0020617, 0.0021994
1: 0.0020382, 0.0026287, 0.0022569, 0.0026135, -0.0004914, 0.0002883
2: 0.0092712, 0.0109130, 0.0094186, 0.0107834, -0.0012605, 0.0011398
3: -0.0049995, -0.0032212, -0.0049394, -0.0035277, -0.0011412, 0.0013601
4: -0.0003801, 0.0013753, -0.0002180, 0.0013102, -0.0012934, 0.0012354
5: 0.0027642, 0.0044569, 0.0028735, 0.0043196, -0.0012488, 0.0012077
6: -0.0111929, -0.0046165, -0.0108992, -0.0051613, -0.0047098, 0.0047918
7: 0.0035344, 0.0126202, 0.0044726, 0.0122871, -0.0067229, 0.0063176
8: 0.9918417, 0.9981200, 0.9923645, 0.9978692, -0.0045971, 0.0044723
9: -0.0141661, -0.0084179, -0.0139530, -0.0089562, -0.0040397, 0.0042372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023423, upper bound: 0.0021265
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023522, upper bound: 0.0021259
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061898, 0.0092971, 0.0063904, 0.0089232, -0.0023323, 0.0024849
1: 0.0018466, 0.0026245, 0.0021438, 0.0026021, -0.0007229, 0.0004032
2: 0.0091548, 0.0109377, 0.0094126, 0.0108268, -0.0014724, 0.0013063
3: -0.0049828, -0.0030242, -0.0048943, -0.0033862, -0.0012712, 0.0016633
4: -0.0004245, 0.0013573, -0.0002762, 0.0012614, -0.0014266, 0.0012734
5: 0.0027190, 0.0044831, 0.0028938, 0.0043656, -0.0013682, 0.0013513
6: -0.0111788, -0.0045129, -0.0107428, -0.0049791, -0.0048871, 0.0052528
7: 0.0032011, 0.0125278, 0.0041136, 0.0120376, -0.0075336, 0.0065778
8: 0.9917424, 0.9980700, 0.9921896, 0.9977021, -0.0050170, 0.0045918
9: -0.0141070, -0.0082648, -0.0137935, -0.0087613, -0.0041687, 0.0046824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0022637
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0023202
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061898, 0.0092971, 0.0063323, 0.0092552, -0.0026798, 0.0024670
1: 0.0018466, 0.0026245, 0.0018112, 0.0026171, -0.0007106, 0.0007373
2: 0.0091548, 0.0109377, 0.0091739, 0.0108589, -0.0014761, 0.0015957
3: -0.0049828, -0.0030242, -0.0049537, -0.0030479, -0.0015417, 0.0016221
4: -0.0004245, 0.0013573, -0.0003418, 0.0013257, -0.0013952, 0.0012532
5: 0.0027190, 0.0044831, 0.0027449, 0.0043996, -0.0013499, 0.0014725
6: -0.0111788, -0.0045129, -0.0110627, -0.0048439, -0.0047428, 0.0052753
7: 0.0032011, 0.0125278, 0.0035861, 0.0123662, -0.0073548, 0.0066674
8: 0.9917424, 0.9980700, 0.9920599, 0.9979573, -0.0049512, 0.0044220
9: -0.0141070, -0.0082648, -0.0140036, -0.0085323, -0.0041170, 0.0045785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0022782
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0023317
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061467, 0.0094144, 0.0063760, 0.0089633, -0.0023839, 0.0026939
1: 0.0017325, 0.0026287, 0.0020952, 0.0026026, -0.0008406, 0.0004664
2: 0.0090599, 0.0109615, 0.0093836, 0.0108347, -0.0016247, 0.0013475
3: -0.0049996, -0.0028977, -0.0048963, -0.0033340, -0.0013626, 0.0017783
4: -0.0004607, 0.0013754, -0.0002895, 0.0012636, -0.0014422, 0.0013258
5: 0.0026696, 0.0045083, 0.0028776, 0.0043740, -0.0014632, 0.0013728
6: -0.0112674, -0.0044127, -0.0107630, -0.0049457, -0.0051034, 0.0052888
7: 0.0029564, 0.0126207, 0.0040167, 0.0120488, -0.0076823, 0.0068757
8: 0.9916462, 0.9981434, 0.9921576, 0.9977137, -0.0050384, 0.0047720
9: -0.0141664, -0.0081420, -0.0138007, -0.0087156, -0.0043424, 0.0047385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0022771
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0023387
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061467, 0.0094144, 0.0063117, 0.0093024, -0.0027756, 0.0026931
1: 0.0017325, 0.0026287, 0.0017582, 0.0026176, -0.0008278, 0.0008013
2: 0.0090599, 0.0109615, 0.0091368, 0.0108703, -0.0016418, 0.0016752
3: -0.0049996, -0.0028977, -0.0049556, -0.0029897, -0.0016363, 0.0017385
4: -0.0004607, 0.0013754, -0.0003581, 0.0013277, -0.0014088, 0.0012992
5: 0.0026696, 0.0045083, 0.0027261, 0.0044117, -0.0014509, 0.0015114
6: -0.0112674, -0.0044127, -0.0110830, -0.0047961, -0.0049383, 0.0053069
7: 0.0029564, 0.0126207, 0.0034766, 0.0123768, -0.0074990, 0.0069389
8: 0.9916462, 0.9981434, 0.9920141, 0.9979684, -0.0049706, 0.0045810
9: -0.0141664, -0.0081420, -0.0140104, -0.0084773, -0.0042695, 0.0046280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0022925
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0023504
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063038, 0.0089297, 0.0063602, 0.0089282, -0.0019013, 0.0018450
1: 0.0022330, 0.0026124, 0.0022412, 0.0026122, -0.0002747, 0.0002665
2: 0.0094229, 0.0108747, 0.0094237, 0.0108435, -0.0010200, 0.0010512
3: -0.0049349, -0.0034334, -0.0049340, -0.0034656, -0.0010550, 0.0010872
4: -0.0003202, 0.0013053, -0.0002852, 0.0013044, -0.0011769, 0.0011421
5: 0.0028781, 0.0044163, 0.0028789, 0.0043833, -0.0010808, 0.0011138
6: -0.0108810, -0.0047777, -0.0108775, -0.0049088, -0.0042882, 0.0044192
7: 0.0039501, 0.0122622, 0.0041287, 0.0122575, -0.0060186, 0.0058401
8: 0.9919963, 0.9978516, 0.9921222, 0.9978483, -0.0042396, 0.0041139
9: -0.0139372, -0.0086221, -0.0139341, -0.0087363, -0.0037343, 0.0038484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022485, upper bound: 0.0021210
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022415, upper bound: 0.0021210
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063038, 0.0089297, 0.0063354, 0.0090067, -0.0020425, 0.0019452
1: 0.0022330, 0.0026124, 0.0022376, 0.0026235, -0.0002951, 0.0002810
2: 0.0094229, 0.0108747, 0.0093803, 0.0108572, -0.0010755, 0.0011292
3: -0.0049349, -0.0034334, -0.0049789, -0.0034514, -0.0011123, 0.0011679
4: -0.0003202, 0.0013053, -0.0003006, 0.0013530, -0.0012643, 0.0012041
5: 0.0028781, 0.0044163, 0.0028329, 0.0043978, -0.0011395, 0.0011965
6: -0.0108810, -0.0047777, -0.0110601, -0.0048512, -0.0045212, 0.0047473
7: 0.0039501, 0.0122622, 0.0040501, 0.0125061, -0.0064653, 0.0061575
8: 0.9919963, 0.9978516, 0.9920669, 0.9980235, -0.0045543, 0.0043375
9: -0.0139372, -0.0086221, -0.0140931, -0.0086861, -0.0039373, 0.0041341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022485, upper bound: 0.0021525
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022415, upper bound: 0.0021525
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062742, 0.0089521, 0.0063486, 0.0089314, -0.0019018, 0.0019183
1: 0.0022287, 0.0026156, 0.0022395, 0.0026126, -0.0002748, 0.0002771
2: 0.0094105, 0.0108910, 0.0094219, 0.0108499, -0.0010606, 0.0010515
3: -0.0049477, -0.0034164, -0.0049359, -0.0034590, -0.0010969, 0.0010875
4: -0.0003385, 0.0013192, -0.0002924, 0.0013064, -0.0011772, 0.0011875
5: 0.0028649, 0.0044337, 0.0028770, 0.0043901, -0.0011237, 0.0011141
6: -0.0109331, -0.0047089, -0.0108851, -0.0048818, -0.0044586, 0.0044203
7: 0.0038564, 0.0123333, 0.0040919, 0.0122678, -0.0060201, 0.0060723
8: 0.9919304, 0.9979017, 0.9920964, 0.9978556, -0.0042407, 0.0042774
9: -0.0139826, -0.0085622, -0.0139407, -0.0087128, -0.0038828, 0.0038494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021306
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022769, upper bound: 0.0021306
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062742, 0.0089521, 0.0063192, 0.0090098, -0.0020430, 0.0020194
1: 0.0022287, 0.0026156, 0.0022352, 0.0026240, -0.0002952, 0.0002917
2: 0.0094105, 0.0108910, 0.0093786, 0.0108661, -0.0011165, 0.0011295
3: -0.0049477, -0.0034164, -0.0049807, -0.0034422, -0.0011547, 0.0011682
4: -0.0003385, 0.0013192, -0.0003106, 0.0013549, -0.0012647, 0.0012500
5: 0.0028649, 0.0044337, 0.0028311, 0.0044072, -0.0011830, 0.0011968
6: -0.0109331, -0.0047089, -0.0110673, -0.0048137, -0.0046936, 0.0047486
7: 0.0038564, 0.0123333, 0.0039991, 0.0125159, -0.0064671, 0.0063923
8: 0.9919304, 0.9979017, 0.9920309, 0.9980304, -0.0045556, 0.0045029
9: -0.0139826, -0.0085622, -0.0140994, -0.0086535, -0.0040874, 0.0041353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021615
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022769, upper bound: 0.0021615
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062336, 0.0089878, 0.0062168, 0.0089984, -0.0020832, 0.0019491
1: 0.0021440, 0.0026124, 0.0021610, 0.0026162, -0.0003809, 0.0003363
2: 0.0093805, 0.0109134, 0.0093775, 0.0109228, -0.0010936, 0.0011613
3: -0.0049349, -0.0033184, -0.0049498, -0.0033262, -0.0011325, 0.0012458
4: -0.0003711, 0.0013053, -0.0003797, 0.0013215, -0.0012659, 0.0011614
5: 0.0028538, 0.0044574, 0.0028449, 0.0044673, -0.0011272, 0.0012115
6: -0.0109073, -0.0046147, -0.0109608, -0.0045755, -0.0043679, 0.0047441
7: 0.0036416, 0.0122623, 0.0036087, 0.0123451, -0.0065273, 0.0059804
8: 0.9918400, 0.9978604, 0.9918025, 0.9979163, -0.0045370, 0.0041707
9: -0.0139372, -0.0084520, -0.0139902, -0.0084245, -0.0038008, 0.0041434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0022790
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0023314
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062336, 0.0089878, 0.0061675, 0.0093072, -0.0024984, 0.0020495
1: 0.0021440, 0.0026124, 0.0018532, 0.0026293, -0.0004003, 0.0006806
2: 0.0093805, 0.0109134, 0.0091526, 0.0109500, -0.0011490, 0.0014587
3: -0.0049349, -0.0033184, -0.0050020, -0.0030221, -0.0014992, 0.0013226
4: -0.0003711, 0.0013053, -0.0004372, 0.0013780, -0.0013491, 0.0012536
5: 0.0028538, 0.0044574, 0.0027075, 0.0044961, -0.0011860, 0.0013994
6: -0.0109073, -0.0046147, -0.0112478, -0.0044610, -0.0046012, 0.0051539
7: 0.0036416, 0.0122623, 0.0031424, 0.0126341, -0.0069524, 0.0066432
8: 0.9918400, 0.9978604, 0.9916925, 0.9981436, -0.0048676, 0.0043945
9: -0.0139372, -0.0084520, -0.0141749, -0.0082238, -0.0041165, 0.0044152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0023555
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0023661
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061939, 0.0090750, 0.0061998, 0.0090350, -0.0021346, 0.0021242
1: 0.0020511, 0.0026156, 0.0021123, 0.0026166, -0.0004829, 0.0004002
2: 0.0093184, 0.0109354, 0.0093505, 0.0109322, -0.0012117, 0.0011986
3: -0.0049477, -0.0032169, -0.0049515, -0.0032740, -0.0012266, 0.0013440
4: -0.0004035, 0.0013193, -0.0003946, 0.0013233, -0.0012811, 0.0012153
5: 0.0028141, 0.0044807, 0.0028291, 0.0044772, -0.0012132, 0.0012343
6: -0.0109852, -0.0045223, -0.0109819, -0.0045360, -0.0045922, 0.0047865
7: 0.0034260, 0.0123335, 0.0035047, 0.0123543, -0.0066609, 0.0062869
8: 0.9917513, 0.9979182, 0.9917645, 0.9979273, -0.0045662, 0.0043588
9: -0.0139827, -0.0083422, -0.0139960, -0.0083737, -0.0039792, 0.0041970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0022864
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0023475
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061939, 0.0090750, 0.0061482, 0.0093571, -0.0025700, 0.0022241
1: 0.0020511, 0.0026156, 0.0018013, 0.0026298, -0.0005018, 0.0007452
2: 0.0093184, 0.0109354, 0.0091161, 0.0109607, -0.0012669, 0.0015224
3: -0.0049477, -0.0032169, -0.0050038, -0.0029650, -0.0015967, 0.0014185
4: -0.0004035, 0.0013193, -0.0004536, 0.0013799, -0.0013617, 0.0013072
5: 0.0028141, 0.0044807, 0.0026892, 0.0045074, -0.0012718, 0.0014265
6: -0.0109852, -0.0045223, -0.0112672, -0.0044162, -0.0048245, 0.0051841
7: 0.0034260, 0.0123335, 0.0030311, 0.0126436, -0.0070735, 0.0069483
8: 0.9917513, 0.9979182, 0.9916496, 0.9981542, -0.0048851, 0.0045816
9: -0.0139827, -0.0083422, -0.0141810, -0.0081681, -0.0042938, 0.0044608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0023640
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0023823
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062713, 0.0090630, 0.0063602, 0.0089282, -0.0019983, 0.0020410
1: 0.0021509, 0.0026245, 0.0022412, 0.0026122, -0.0003744, 0.0002850
2: 0.0093401, 0.0108927, 0.0094237, 0.0108435, -0.0011418, 0.0011048
3: -0.0049828, -0.0033385, -0.0049340, -0.0034656, -0.0011281, 0.0012262
4: -0.0003480, 0.0013572, -0.0002852, 0.0013044, -0.0012454, 0.0012212
5: 0.0028084, 0.0044354, 0.0028789, 0.0043833, -0.0011834, 0.0011706
6: -0.0110978, -0.0047021, -0.0108775, -0.0049088, -0.0046109, 0.0046446
7: 0.0037583, 0.0125275, 0.0041287, 0.0122575, -0.0064220, 0.0062449
8: 0.9919239, 0.9980459, 0.9921222, 0.9978483, -0.0044559, 0.0044074
9: -0.0141068, -0.0085273, -0.0139341, -0.0087363, -0.0039932, 0.0040762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023032, upper bound: 0.0021206
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023019, upper bound: 0.0021206
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062713, 0.0090630, 0.0063354, 0.0090067, -0.0019416, 0.0019645
1: 0.0021509, 0.0026245, 0.0022376, 0.0026235, -0.0003639, 0.0002723
2: 0.0093401, 0.0108927, 0.0093803, 0.0108572, -0.0011017, 0.0010735
3: -0.0049828, -0.0033385, -0.0049789, -0.0034514, -0.0010777, 0.0011904
4: -0.0003480, 0.0013572, -0.0003006, 0.0013530, -0.0012098, 0.0011666
5: 0.0028084, 0.0044354, 0.0028329, 0.0043978, -0.0011369, 0.0011374
6: -0.0110978, -0.0047021, -0.0110601, -0.0048512, -0.0044127, 0.0045129
7: 0.0037583, 0.0125275, 0.0040501, 0.0125061, -0.0062375, 0.0059658
8: 0.9919239, 0.9980459, 0.9920669, 0.9980235, -0.0043295, 0.0042130
9: -0.0141068, -0.0085273, -0.0140931, -0.0086861, -0.0038147, 0.0039595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023032, upper bound: 0.0021281
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023019, upper bound: 0.0021277
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062344, 0.0091598, 0.0063486, 0.0089314, -0.0019971, 0.0022246
1: 0.0020382, 0.0026287, 0.0022395, 0.0026126, -0.0004920, 0.0002965
2: 0.0092712, 0.0109130, 0.0094219, 0.0108499, -0.0012661, 0.0011041
3: -0.0049995, -0.0032212, -0.0049359, -0.0034590, -0.0011735, 0.0013354
4: -0.0003801, 0.0013753, -0.0002924, 0.0013064, -0.0012555, 0.0012704
5: 0.0027642, 0.0044569, 0.0028770, 0.0043901, -0.0012734, 0.0011699
6: -0.0111929, -0.0046165, -0.0108851, -0.0048818, -0.0048340, 0.0046418
7: 0.0035344, 0.0126202, 0.0040919, 0.0122678, -0.0065435, 0.0064962
8: 0.9918417, 0.9981200, 0.9920964, 0.9978556, -0.0044531, 0.0045963
9: -0.0141661, -0.0084179, -0.0139407, -0.0087128, -0.0041539, 0.0041146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023443, upper bound: 0.0021305
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023538, upper bound: 0.0021305
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062344, 0.0091598, 0.0063192, 0.0090098, -0.0019421, 0.0021622
1: 0.0020382, 0.0026287, 0.0022352, 0.0026240, -0.0004804, 0.0002830
2: 0.0092712, 0.0109130, 0.0093786, 0.0108661, -0.0012343, 0.0010737
3: -0.0049995, -0.0032212, -0.0049807, -0.0034422, -0.0011200, 0.0012984
4: -0.0003801, 0.0013753, -0.0003106, 0.0013549, -0.0012208, 0.0012124
5: 0.0027642, 0.0044569, 0.0028311, 0.0044072, -0.0012303, 0.0011377
6: -0.0111929, -0.0046165, -0.0110673, -0.0048137, -0.0046236, 0.0045139
7: 0.0035344, 0.0126202, 0.0039991, 0.0125159, -0.0063609, 0.0062000
8: 0.9918417, 0.9981200, 0.9920309, 0.9980304, -0.0043305, 0.0043894
9: -0.0141661, -0.0084179, -0.0140994, -0.0086535, -0.0039645, 0.0040006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023443, upper bound: 0.0021376
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023538, upper bound: 0.0021372
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061898, 0.0092971, 0.0062168, 0.0089984, -0.0021755, 0.0024091
1: 0.0018466, 0.0026245, 0.0021610, 0.0026162, -0.0007144, 0.0003564
2: 0.0091548, 0.0109377, 0.0093775, 0.0109228, -0.0014287, 0.0012124
3: -0.0049828, -0.0030242, -0.0049498, -0.0033262, -0.0012120, 0.0015981
4: -0.0004245, 0.0013573, -0.0003797, 0.0013215, -0.0013522, 0.0012476
5: 0.0027190, 0.0044831, 0.0028449, 0.0044673, -0.0013336, 0.0012656
6: -0.0111788, -0.0045129, -0.0109608, -0.0045755, -0.0048045, 0.0049586
7: 0.0032011, 0.0125278, 0.0036087, 0.0123451, -0.0071547, 0.0064209
8: 0.9917424, 0.9980700, 0.9918025, 0.9979163, -0.0047429, 0.0045122
9: -0.0141070, -0.0082648, -0.0139902, -0.0084245, -0.0040825, 0.0044394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0022720
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0023314
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061898, 0.0092971, 0.0061675, 0.0093072, -0.0024630, 0.0024006
1: 0.0018466, 0.0026245, 0.0018532, 0.0026293, -0.0007015, 0.0006682
2: 0.0091548, 0.0109377, 0.0091526, 0.0109500, -0.0014425, 0.0014550
3: -0.0049828, -0.0030242, -0.0050020, -0.0030221, -0.0014594, 0.0015604
4: -0.0004245, 0.0013573, -0.0004372, 0.0013780, -0.0013195, 0.0012199
5: 0.0027190, 0.0044831, 0.0027075, 0.0044961, -0.0013104, 0.0013667
6: -0.0111788, -0.0045129, -0.0112478, -0.0044610, -0.0046150, 0.0049599
7: 0.0032011, 0.0125278, 0.0031424, 0.0126341, -0.0069812, 0.0064641
8: 0.9917424, 0.9980700, 0.9916925, 0.9981436, -0.0046660, 0.0043108
9: -0.0141070, -0.0082648, -0.0141749, -0.0082238, -0.0040059, 0.0043320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0022894
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0023423
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061467, 0.0094144, 0.0061998, 0.0090350, -0.0022248, 0.0026162
1: 0.0017325, 0.0026287, 0.0021123, 0.0026166, -0.0008324, 0.0004205
2: 0.0090599, 0.0109615, 0.0093505, 0.0109322, -0.0015822, 0.0012484
3: -0.0049996, -0.0028977, -0.0049515, -0.0032740, -0.0013068, 0.0017193
4: -0.0004607, 0.0013754, -0.0003946, 0.0013233, -0.0013678, 0.0013022
5: 0.0026696, 0.0045083, 0.0028291, 0.0044772, -0.0014273, 0.0012871
6: -0.0112674, -0.0044127, -0.0109819, -0.0045360, -0.0050266, 0.0049961
7: 0.0029564, 0.0126207, 0.0035047, 0.0123543, -0.0073053, 0.0067312
8: 0.9916462, 0.9981434, 0.9917645, 0.9979273, -0.0047672, 0.0047000
9: -0.0141664, -0.0081420, -0.0139960, -0.0083737, -0.0042633, 0.0044955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0022785
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0023475
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061467, 0.0094144, 0.0061482, 0.0093571, -0.0025573, 0.0026257
1: 0.0017325, 0.0026287, 0.0018013, 0.0026298, -0.0008199, 0.0007333
2: 0.0090599, 0.0109615, 0.0091161, 0.0109607, -0.0016132, 0.0015316
3: -0.0049996, -0.0028977, -0.0050038, -0.0029650, -0.0015588, 0.0016813
4: -0.0004607, 0.0013754, -0.0004536, 0.0013799, -0.0013370, 0.0012731
5: 0.0026696, 0.0045083, 0.0026892, 0.0045074, -0.0014075, 0.0014029
6: -0.0112674, -0.0044127, -0.0112672, -0.0044162, -0.0048410, 0.0050005
7: 0.0029564, 0.0126207, 0.0030311, 0.0126436, -0.0071405, 0.0067705
8: 0.9916462, 0.9981434, 0.9916496, 0.9981542, -0.0046945, 0.0045008
9: -0.0141664, -0.0081420, -0.0141810, -0.0081681, -0.0041820, 0.0043941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0022966
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0023600
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.60 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0021081
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0021081
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0021406
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0021406
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0021211
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0021210
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0021513
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0021513
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0022712
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023202
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023475
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023525
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0022856
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023388
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023627
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023712
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021077
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021077
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021128
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021126
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022447, upper bound: 0.0021210
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022464, upper bound: 0.0021208
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022447, upper bound: 0.0021265
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022464, upper bound: 0.0021259
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0022637
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023202
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0022782
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023317
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0022773
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023388
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0022925
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023504
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0021759
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0021758
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021425, upper bound: 0.0022167
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021340, upper bound: 0.0022163
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0021896
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0021896
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021710, upper bound: 0.0022299
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021638, upper bound: 0.0022299
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023804
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0023827
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0024529
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0024258
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023893
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0023996
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0024622
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022367, upper bound: 0.0024439
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021750
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021750
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022006, upper bound: 0.0021796
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021794
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022445, upper bound: 0.0021889
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022456, upper bound: 0.0021888
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022445, upper bound: 0.0021945
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022456, upper bound: 0.0021945
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023738
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023815
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023876
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0023921
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023832
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023991
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0023993
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022684, upper bound: 0.0024089
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022423, upper bound: 0.0021081
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021081
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022423, upper bound: 0.0021406
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021406
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022747, upper bound: 0.0021211
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021210
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022747, upper bound: 0.0021513
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021513
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0022704
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0023202
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0023468
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022822, upper bound: 0.0023525
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0022848
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0023388
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0023623
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022945, upper bound: 0.0023712
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022983, upper bound: 0.0021077
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022981, upper bound: 0.0021077
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022983, upper bound: 0.0021128
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022981, upper bound: 0.0021126
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023423, upper bound: 0.0021210
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023522, upper bound: 0.0021208
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023423, upper bound: 0.0021265
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023522, upper bound: 0.0021259
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0022637
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0023202
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0022782
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023132, upper bound: 0.0023317
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0022771
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0023387
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0022925
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023395, upper bound: 0.0023504
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022485, upper bound: 0.0021210
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022415, upper bound: 0.0021210
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022485, upper bound: 0.0021525
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022415, upper bound: 0.0021525
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021306
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022769, upper bound: 0.0021306
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022758, upper bound: 0.0021615
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022769, upper bound: 0.0021615
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0022790
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0023314
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0023555
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022840, upper bound: 0.0023661
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0022864
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0023475
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0023640
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0022948, upper bound: 0.0023823
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023032, upper bound: 0.0021206
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023019, upper bound: 0.0021206
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023032, upper bound: 0.0021281
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023019, upper bound: 0.0021277
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023443, upper bound: 0.0021305
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023538, upper bound: 0.0021305
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023443, upper bound: 0.0021376
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023538, upper bound: 0.0021372
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0022720
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0023314
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0022894
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023150, upper bound: 0.0023423
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0022785
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0023475
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0022966
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 8, lower bound: -0.0023398, upper bound: 0.0023600

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0065681, 0.0088262, 0.0063904, 0.0089232, -0.0018573, 0.0019026
1: 0.0022712, 0.0025974, 0.0021438, 0.0026021, -0.0002553, 0.0003841
2: 0.0094801, 0.0107285, 0.0094126, 0.0108268, -0.0010519, 0.0010449
3: -0.0048757, -0.0035845, -0.0048943, -0.0033862, -0.0011912, 0.0010103
4: -0.0001565, 0.0012413, -0.0002762, 0.0012614, -0.0010937, 0.0011879
5: 0.0029387, 0.0042615, 0.0028938, 0.0043656, -0.0011145, 0.0010719
6: -0.0106406, -0.0053921, -0.0107428, -0.0049791, -0.0044221, 0.0041428
7: 0.0047869, 0.0119349, 0.0041136, 0.0120376, -0.0055927, 0.0061391
8: 0.9925859, 0.9976211, 0.9921896, 0.9977021, -0.0039510, 0.0042424
9: -0.0137278, -0.0091572, -0.0137935, -0.0087613, -0.0038891, 0.0035761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0021643
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0021577
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064450, 0.0088356, 0.0063904, 0.0089232, -0.0018851, 0.0018077
1: 0.0022534, 0.0025988, 0.0021438, 0.0026021, -0.0002592, 0.0003697
2: 0.0094749, 0.0107966, 0.0094126, 0.0108268, -0.0009994, 0.0010591
3: -0.0048811, -0.0035141, -0.0048943, -0.0033862, -0.0011377, 0.0010260
4: -0.0002327, 0.0012471, -0.0002762, 0.0012614, -0.0011107, 0.0011291
5: 0.0029332, 0.0043336, 0.0028938, 0.0043656, -0.0010589, 0.0010890
6: -0.0106624, -0.0051061, -0.0107428, -0.0049791, -0.0042015, 0.0042109
7: 0.0043973, 0.0119646, 0.0041136, 0.0120376, -0.0056799, 0.0058396
8: 0.9923114, 0.9976419, 0.9921896, 0.9977021, -0.0040140, 0.0040308
9: -0.0137468, -0.0089081, -0.0137935, -0.0087613, -0.0036968, 0.0036319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0021875
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0021827
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065681, 0.0088262, 0.0063323, 0.0092552, -0.0023131, 0.0019934
1: 0.0022712, 0.0025974, 0.0018112, 0.0026171, -0.0002751, 0.0007484
2: 0.0094801, 0.0107285, 0.0091739, 0.0108589, -0.0011021, 0.0013794
3: -0.0048757, -0.0035845, -0.0049537, -0.0030479, -0.0015750, 0.0010887
4: -0.0001565, 0.0012413, -0.0003418, 0.0013257, -0.0011785, 0.0012761
5: 0.0029387, 0.0042615, 0.0027449, 0.0043996, -0.0011677, 0.0012758
6: -0.0106406, -0.0053921, -0.0110627, -0.0048439, -0.0046332, 0.0045672
7: 0.0047869, 0.0119349, 0.0035861, 0.0123662, -0.0060267, 0.0067966
8: 0.9925859, 0.9976211, 0.9920599, 0.9979573, -0.0042884, 0.0044449
9: -0.0137278, -0.0091572, -0.0140036, -0.0085323, -0.0041929, 0.0038536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0022354
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0022356
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064450, 0.0088356, 0.0063323, 0.0092552, -0.0023624, 0.0019061
1: 0.0022534, 0.0025988, 0.0018112, 0.0026171, -0.0002795, 0.0007353
2: 0.0094749, 0.0107966, 0.0091739, 0.0108589, -0.0010538, 0.0014149
3: -0.0048811, -0.0035141, -0.0049537, -0.0030479, -0.0015243, 0.0011064
4: -0.0002327, 0.0012471, -0.0003418, 0.0013257, -0.0011977, 0.0012215
5: 0.0029332, 0.0043336, 0.0027449, 0.0043996, -0.0011166, 0.0012973
6: -0.0106624, -0.0051061, -0.0110627, -0.0048439, -0.0044302, 0.0046429
7: 0.0043973, 0.0119646, 0.0035861, 0.0123662, -0.0061249, 0.0065135
8: 0.9923114, 0.9976419, 0.9920599, 0.9979573, -0.0043588, 0.0042502
9: -0.0137468, -0.0089081, -0.0140036, -0.0085323, -0.0040139, 0.0039164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0022276
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0022239
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0065371, 0.0088472, 0.0063760, 0.0089633, -0.0019047, 0.0019803
1: 0.0022667, 0.0026005, 0.0020952, 0.0026026, -0.0002561, 0.0004475
2: 0.0094685, 0.0107457, 0.0093836, 0.0108347, -0.0010949, 0.0010825
3: -0.0048877, -0.0035668, -0.0048963, -0.0033340, -0.0012838, 0.0010138
4: -0.0001757, 0.0012543, -0.0002895, 0.0012636, -0.0010975, 0.0012405
5: 0.0029263, 0.0042796, 0.0028776, 0.0043740, -0.0011601, 0.0010920
6: -0.0106894, -0.0053200, -0.0107630, -0.0049457, -0.0046027, 0.0041749
7: 0.0046886, 0.0120014, 0.0040167, 0.0120488, -0.0056121, 0.0064388
8: 0.9925166, 0.9976679, 0.9921576, 0.9977137, -0.0039698, 0.0044157
9: -0.0137704, -0.0090944, -0.0138007, -0.0087156, -0.0040632, 0.0035885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021219, upper bound: 0.0021779
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021218, upper bound: 0.0021704
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064118, 0.0089004, 0.0063760, 0.0089633, -0.0019429, 0.0019512
1: 0.0021736, 0.0026016, 0.0020952, 0.0026026, -0.0003401, 0.0004330
2: 0.0094296, 0.0108149, 0.0093836, 0.0108347, -0.0010908, 0.0011045
3: -0.0048921, -0.0034215, -0.0048963, -0.0033340, -0.0012295, 0.0011078
4: -0.0002606, 0.0012590, -0.0002895, 0.0012636, -0.0011224, 0.0011817
5: 0.0029032, 0.0043530, 0.0028776, 0.0043740, -0.0011319, 0.0011121
6: -0.0107270, -0.0050288, -0.0107630, -0.0049457, -0.0044115, 0.0042382
7: 0.0042078, 0.0120254, 0.0040167, 0.0120488, -0.0057887, 0.0061393
8: 0.9922373, 0.9976914, 0.9921576, 0.9977137, -0.0040329, 0.0042130
9: -0.0137857, -0.0088135, -0.0138007, -0.0087156, -0.0038711, 0.0036737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021219, upper bound: 0.0022033
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021218, upper bound: 0.0021972
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065371, 0.0088472, 0.0063117, 0.0093024, -0.0023863, 0.0020732
1: 0.0022667, 0.0026005, 0.0017582, 0.0026176, -0.0002751, 0.0008135
2: 0.0094685, 0.0107457, 0.0091368, 0.0108703, -0.0011462, 0.0014441
3: -0.0048877, -0.0035668, -0.0049556, -0.0029897, -0.0016748, 0.0010888
4: -0.0001757, 0.0012543, -0.0003581, 0.0013277, -0.0011787, 0.0013302
5: 0.0029263, 0.0042796, 0.0027261, 0.0044117, -0.0012145, 0.0012995
6: -0.0106894, -0.0053200, -0.0110830, -0.0047961, -0.0048188, 0.0045798
7: 0.0046886, 0.0120014, 0.0034766, 0.0123768, -0.0060277, 0.0071042
8: 0.9925166, 0.9976679, 0.9920141, 0.9979684, -0.0042923, 0.0046230
9: -0.0137704, -0.0090944, -0.0140104, -0.0084773, -0.0043717, 0.0038543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021211, upper bound: 0.0022447
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021209, upper bound: 0.0022464
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064118, 0.0089004, 0.0063117, 0.0093024, -0.0024531, 0.0020494
1: 0.0021736, 0.0026016, 0.0017582, 0.0026176, -0.0003602, 0.0008001
2: 0.0094296, 0.0108149, 0.0091368, 0.0108703, -0.0011451, 0.0014922
3: -0.0048921, -0.0034215, -0.0049556, -0.0029897, -0.0016222, 0.0011875
4: -0.0002606, 0.0012590, -0.0003581, 0.0013277, -0.0012087, 0.0012741
5: 0.0029032, 0.0043530, 0.0027261, 0.0044117, -0.0011895, 0.0013314
6: -0.0107270, -0.0050288, -0.0110830, -0.0047961, -0.0046398, 0.0046766
7: 0.0042078, 0.0120254, 0.0034766, 0.0123768, -0.0062300, 0.0068144
8: 0.9922373, 0.9976914, 0.9920141, 0.9979684, -0.0043786, 0.0044320
9: -0.0137857, -0.0088135, -0.0140104, -0.0084773, -0.0041884, 0.0039559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021211, upper bound: 0.0022420
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021209, upper bound: 0.0022367
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0064294, 0.0090742, 0.0065077, 0.0088541, -0.0018852, 0.0021643
1: 0.0020481, 0.0026165, 0.0022625, 0.0026015, -0.0004908, 0.0002863
2: 0.0093169, 0.0108052, 0.0094647, 0.0107619, -0.0012362, 0.0010423
3: -0.0049510, -0.0033148, -0.0048916, -0.0035500, -0.0011330, 0.0012848
4: -0.0002613, 0.0013228, -0.0001939, 0.0012585, -0.0011872, 0.0012265
5: 0.0028140, 0.0043427, 0.0029224, 0.0042968, -0.0012347, 0.0011043
6: -0.0109958, -0.0050698, -0.0107053, -0.0052517, -0.0046692, 0.0043817
7: 0.0041304, 0.0123517, 0.0045956, 0.0120230, -0.0062006, 0.0062720
8: 0.9922765, 0.9979306, 0.9924511, 0.9976831, -0.0042037, 0.0044390
9: -0.0139943, -0.0088056, -0.0137842, -0.0090349, -0.0040105, 0.0038916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020948, upper bound: 0.0019471
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0019317
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064968, 0.0090959, 0.0065500, 0.0088528, -0.0018943, 0.0021379
1: 0.0021688, 0.0026293, 0.0022686, 0.0026013, -0.0003738, 0.0002982
2: 0.0093221, 0.0107679, 0.0094654, 0.0107385, -0.0011973, 0.0010473
3: -0.0050020, -0.0034540, -0.0048909, -0.0035742, -0.0011802, 0.0011785
4: -0.0002096, 0.0013780, -0.0001677, 0.0012578, -0.0011822, 0.0012776
5: 0.0027890, 0.0043032, 0.0029231, 0.0042721, -0.0012386, 0.0011097
6: -0.0111745, -0.0052265, -0.0107024, -0.0053500, -0.0048249, 0.0044030
7: 0.0044584, 0.0126339, 0.0047296, 0.0120190, -0.0061066, 0.0065332
8: 0.9924269, 0.9981201, 0.9925454, 0.9976803, -0.0042240, 0.0046110
9: -0.0141748, -0.0089793, -0.0137816, -0.0091206, -0.0041775, 0.0038703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020827, upper bound: 0.0019415
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0019251
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0064294, 0.0090742, 0.0064725, 0.0089374, -0.0018282, 0.0021018
1: 0.0020481, 0.0026165, 0.0022574, 0.0026135, -0.0004792, 0.0002723
2: 0.0093169, 0.0108052, 0.0094186, 0.0107814, -0.0012054, 0.0010107
3: -0.0049510, -0.0033148, -0.0049393, -0.0035298, -0.0010778, 0.0012463
4: -0.0002613, 0.0013228, -0.0002157, 0.0013101, -0.0011515, 0.0011667
5: 0.0028140, 0.0043427, 0.0028735, 0.0043175, -0.0011924, 0.0010709
6: -0.0109958, -0.0050698, -0.0108990, -0.0051698, -0.0044527, 0.0042491
7: 0.0041304, 0.0123517, 0.0044841, 0.0122868, -0.0060153, 0.0059663
8: 0.9922765, 0.9979306, 0.9923726, 0.9978690, -0.0040765, 0.0042254
9: -0.0139943, -0.0088056, -0.0139529, -0.0089636, -0.0038150, 0.0037748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020961, upper bound: 0.0019498
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020796, upper bound: 0.0019337
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064968, 0.0090959, 0.0065168, 0.0089362, -0.0018395, 0.0020507
1: 0.0021688, 0.0026293, 0.0022638, 0.0026133, -0.0003628, 0.0002839
2: 0.0093221, 0.0107679, 0.0094193, 0.0107569, -0.0011509, 0.0010170
3: -0.0050020, -0.0034540, -0.0049386, -0.0035552, -0.0011235, 0.0011460
4: -0.0002096, 0.0013780, -0.0001883, 0.0013094, -0.0011479, 0.0012162
5: 0.0027890, 0.0043032, 0.0028743, 0.0042915, -0.0011856, 0.0010776
6: -0.0111745, -0.0052265, -0.0108961, -0.0052728, -0.0046025, 0.0042756
7: 0.0044584, 0.0126339, 0.0046244, 0.0122828, -0.0059296, 0.0062194
8: 0.9924269, 0.9981201, 0.9924713, 0.9978662, -0.0041018, 0.0043928
9: -0.0141748, -0.0089793, -0.0139503, -0.0090533, -0.0039768, 0.0037579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020827, upper bound: 0.0019433
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0019255
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0065415, 0.0089011, 0.0063904, 0.0089232, -0.0019585, 0.0020400
1: 0.0022674, 0.0026083, 0.0021438, 0.0026021, -0.0002699, 0.0004040
2: 0.0094387, 0.0107432, 0.0094126, 0.0108268, -0.0011278, 0.0011008
3: -0.0049185, -0.0035693, -0.0048943, -0.0033862, -0.0012698, 0.0010681
4: -0.0001730, 0.0012877, -0.0002762, 0.0012614, -0.0011563, 0.0012729
5: 0.0028948, 0.0042770, 0.0028938, 0.0043656, -0.0011950, 0.0011312
6: -0.0108146, -0.0053304, -0.0107428, -0.0049791, -0.0047414, 0.0043780
7: 0.0047028, 0.0121719, 0.0041136, 0.0120376, -0.0059131, 0.0065740
8: 0.9925266, 0.9977881, 0.9921896, 0.9977021, -0.0041767, 0.0045487
9: -0.0138794, -0.0091034, -0.0137935, -0.0087613, -0.0041672, 0.0037810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0021566
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0021508
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063911, 0.0091184, 0.0063904, 0.0089232, -0.0019834, 0.0022104
1: 0.0019597, 0.0026122, 0.0021438, 0.0026021, -0.0005810, 0.0003895
2: 0.0092774, 0.0108264, 0.0094126, 0.0108268, -0.0012803, 0.0011135
3: -0.0049340, -0.0032132, -0.0048943, -0.0033862, -0.0012161, 0.0013696
4: -0.0002929, 0.0013044, -0.0002762, 0.0012614, -0.0011995, 0.0012139
5: 0.0028056, 0.0043652, 0.0028938, 0.0043656, -0.0012427, 0.0011466
6: -0.0109475, -0.0049806, -0.0107428, -0.0049791, -0.0046218, 0.0044395
7: 0.0039167, 0.0122577, 0.0041136, 0.0120376, -0.0063131, 0.0062733
8: 0.9921910, 0.9978695, 0.9921896, 0.9977021, -0.0042333, 0.0043679
9: -0.0139342, -0.0086982, -0.0137935, -0.0087613, -0.0039741, 0.0039355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0021871
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0021826
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065415, 0.0089011, 0.0063323, 0.0092552, -0.0022929, 0.0019415
1: 0.0022674, 0.0026083, 0.0018112, 0.0026171, -0.0002611, 0.0007350
2: 0.0094387, 0.0107432, 0.0091739, 0.0108589, -0.0010734, 0.0013817
3: -0.0049185, -0.0035693, -0.0049537, -0.0030479, -0.0015350, 0.0010333
4: -0.0001730, 0.0012877, -0.0003418, 0.0013257, -0.0011186, 0.0012422
5: 0.0028948, 0.0042770, 0.0027449, 0.0043996, -0.0011373, 0.0012459
6: -0.0108146, -0.0053304, -0.0110627, -0.0048439, -0.0045127, 0.0043670
7: 0.0047028, 0.0121719, 0.0035861, 0.0123662, -0.0057202, 0.0066167
8: 0.9925266, 0.9977881, 0.9920599, 0.9979573, -0.0040776, 0.0043293
9: -0.0138794, -0.0091034, -0.0140036, -0.0085323, -0.0040813, 0.0036576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021103, upper bound: 0.0021680
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021102, upper bound: 0.0021616
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063911, 0.0091184, 0.0063323, 0.0092552, -0.0023604, 0.0021788
1: 0.0019597, 0.0026122, 0.0018112, 0.0026171, -0.0005686, 0.0007223
2: 0.0092774, 0.0108264, 0.0091739, 0.0108589, -0.0012802, 0.0014408
3: -0.0049340, -0.0032132, -0.0049537, -0.0030479, -0.0014829, 0.0013319
4: -0.0002929, 0.0013044, -0.0003418, 0.0013257, -0.0011678, 0.0011890
5: 0.0028056, 0.0043652, 0.0027449, 0.0043996, -0.0012124, 0.0012760
6: -0.0109475, -0.0049806, -0.0110627, -0.0048439, -0.0044264, 0.0044529
7: 0.0039167, 0.0122577, 0.0035861, 0.0123662, -0.0061438, 0.0063392
8: 0.9921910, 0.9978695, 0.9920599, 0.9979573, -0.0041614, 0.0041706
9: -0.0139342, -0.0086982, -0.0140036, -0.0085323, -0.0039073, 0.0038315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021103, upper bound: 0.0022133
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021102, upper bound: 0.0022089
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0065007, 0.0089304, 0.0063760, 0.0089633, -0.0020037, 0.0021199
1: 0.0022615, 0.0026125, 0.0020952, 0.0026026, -0.0002704, 0.0004677
2: 0.0094225, 0.0107658, 0.0093836, 0.0108347, -0.0011721, 0.0011373
3: -0.0049353, -0.0035460, -0.0048963, -0.0033340, -0.0013636, 0.0010704
4: -0.0001982, 0.0013058, -0.0002895, 0.0012636, -0.0011587, 0.0013269
5: 0.0028776, 0.0043009, 0.0028776, 0.0043740, -0.0012419, 0.0011500
6: -0.0108827, -0.0052355, -0.0107630, -0.0049457, -0.0049273, 0.0044049
7: 0.0045736, 0.0122646, 0.0040167, 0.0120488, -0.0059254, 0.0068809
8: 0.9924356, 0.9978532, 0.9921576, 0.9977137, -0.0041905, 0.0047271
9: -0.0139386, -0.0090208, -0.0138007, -0.0087156, -0.0043459, 0.0037888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021513, upper bound: 0.0021710
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021513, upper bound: 0.0021638
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063456, 0.0092234, 0.0063760, 0.0089633, -0.0020402, 0.0023990
1: 0.0018448, 0.0026165, 0.0020952, 0.0026026, -0.0006997, 0.0004532
2: 0.0091992, 0.0108516, 0.0093836, 0.0108347, -0.0014202, 0.0011583
3: -0.0049513, -0.0030849, -0.0048963, -0.0033340, -0.0013095, 0.0014879
4: -0.0003301, 0.0013231, -0.0002895, 0.0012636, -0.0012137, 0.0012683
5: 0.0027580, 0.0043918, 0.0028776, 0.0043740, -0.0013307, 0.0011692
6: -0.0110447, -0.0048749, -0.0107630, -0.0049457, -0.0048401, 0.0044645
7: 0.0036686, 0.0123530, 0.0040167, 0.0120488, -0.0064536, 0.0065820
8: 0.9920896, 0.9979456, 0.9921576, 0.9977137, -0.0042499, 0.0045559
9: -0.0139952, -0.0085727, -0.0138007, -0.0087156, -0.0041542, 0.0039870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021513, upper bound: 0.0022033
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021513, upper bound: 0.0021972
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065007, 0.0089304, 0.0063117, 0.0093024, -0.0023834, 0.0020192
1: 0.0022615, 0.0026125, 0.0017582, 0.0026176, -0.0002619, 0.0008005
2: 0.0094225, 0.0107658, 0.0091368, 0.0108703, -0.0011164, 0.0014584
3: -0.0049353, -0.0035460, -0.0049556, -0.0029897, -0.0016344, 0.0010365
4: -0.0001982, 0.0013058, -0.0003581, 0.0013277, -0.0011221, 0.0012953
5: 0.0028776, 0.0043009, 0.0027261, 0.0044117, -0.0011829, 0.0012817
6: -0.0108827, -0.0052355, -0.0110830, -0.0047961, -0.0046933, 0.0043955
7: 0.0045736, 0.0122646, 0.0034766, 0.0123768, -0.0057380, 0.0069221
8: 0.9924356, 0.9978532, 0.9920141, 0.9979684, -0.0040962, 0.0045025
9: -0.0139386, -0.0090208, -0.0140104, -0.0084773, -0.0042573, 0.0036690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021544, upper bound: 0.0021802
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021543, upper bound: 0.0021753
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063456, 0.0092234, 0.0063117, 0.0093024, -0.0024821, 0.0023879
1: 0.0018448, 0.0026165, 0.0017582, 0.0026176, -0.0006875, 0.0007878
2: 0.0091992, 0.0108516, 0.0091368, 0.0108703, -0.0014402, 0.0015331
3: -0.0049513, -0.0030849, -0.0049556, -0.0029897, -0.0015837, 0.0014481
4: -0.0003301, 0.0013231, -0.0003581, 0.0013277, -0.0011818, 0.0012417
5: 0.0027580, 0.0043918, 0.0027261, 0.0044117, -0.0013071, 0.0013214
6: -0.0110447, -0.0048749, -0.0110830, -0.0047961, -0.0046483, 0.0044883
7: 0.0036686, 0.0123530, 0.0034766, 0.0123768, -0.0062829, 0.0066446
8: 0.9920896, 0.9979456, 0.9920141, 0.9979684, -0.0041755, 0.0043580
9: -0.0139952, -0.0085727, -0.0140104, -0.0084773, -0.0040818, 0.0038821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021544, upper bound: 0.0022278
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021543, upper bound: 0.0022208
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065089, 0.0088356, 0.0063393, 0.0090066, -0.0020252, 0.0020455
1: 0.0022627, 0.0025988, 0.0022381, 0.0026235, -0.0002926, 0.0002955
2: 0.0094749, 0.0107612, 0.0093803, 0.0108550, -0.0011309, 0.0011197
3: -0.0048811, -0.0035507, -0.0049789, -0.0034537, -0.0011696, 0.0011580
4: -0.0001931, 0.0012471, -0.0002982, 0.0013530, -0.0012536, 0.0012662
5: 0.0029332, 0.0042961, 0.0028330, 0.0043955, -0.0011983, 0.0011864
6: -0.0106623, -0.0052546, -0.0110599, -0.0048602, -0.0047543, 0.0047072
7: 0.0045996, 0.0119645, 0.0040625, 0.0125059, -0.0064108, 0.0064750
8: 0.9924539, 0.9976419, 0.9920755, 0.9980233, -0.0045159, 0.0045611
9: -0.0137468, -0.0090374, -0.0140930, -0.0086940, -0.0041403, 0.0040992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020177, upper bound: 0.0020553
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020038, upper bound: 0.0020519
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0065732, 0.0089212, 0.0063791, 0.0090053, -0.0020374, 0.0021169
1: 0.0022719, 0.0026112, 0.0022439, 0.0026233, -0.0002943, 0.0003058
2: 0.0094276, 0.0107257, 0.0093811, 0.0108330, -0.0011704, 0.0011264
3: -0.0049300, -0.0035874, -0.0049781, -0.0034764, -0.0012105, 0.0011650
4: -0.0001534, 0.0013001, -0.0002735, 0.0013521, -0.0012612, 0.0013104
5: 0.0028830, 0.0042585, 0.0028338, 0.0043722, -0.0012401, 0.0011935
6: -0.0108613, -0.0054039, -0.0110567, -0.0049528, -0.0049202, 0.0047354
7: 0.0048030, 0.0122355, 0.0041886, 0.0125016, -0.0064492, 0.0067009
8: 0.9925972, 0.9978328, 0.9921645, 0.9980203, -0.0045430, 0.0047203
9: -0.0139201, -0.0091675, -0.0140902, -0.0087747, -0.0042847, 0.0041238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019998, upper bound: 0.0020365
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019750, upper bound: 0.0020309
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0064781, 0.0088545, 0.0063231, 0.0090097, -0.0020293, 0.0021196
1: 0.0022582, 0.0026015, 0.0022358, 0.0026239, -0.0002932, 0.0003062
2: 0.0094645, 0.0107783, 0.0093786, 0.0108640, -0.0011719, 0.0011219
3: -0.0048919, -0.0035331, -0.0049806, -0.0034444, -0.0012120, 0.0011604
4: -0.0002122, 0.0012588, -0.0003082, 0.0013549, -0.0012562, 0.0013121
5: 0.0029221, 0.0043142, 0.0028312, 0.0044050, -0.0012417, 0.0011887
6: -0.0107062, -0.0051830, -0.0110671, -0.0048227, -0.0049266, 0.0047166
7: 0.0045020, 0.0120242, 0.0040114, 0.0125157, -0.0064236, 0.0067096
8: 0.9923853, 0.9976839, 0.9920396, 0.9980302, -0.0045249, 0.0047264
9: -0.0137849, -0.0089751, -0.0140992, -0.0086613, -0.0042903, 0.0041074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020256, upper bound: 0.0020573
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020148, upper bound: 0.0020528
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0065433, 0.0089446, 0.0063623, 0.0090083, -0.0020401, 0.0021984
1: 0.0022676, 0.0026145, 0.0022415, 0.0026237, -0.0002947, 0.0003176
2: 0.0094146, 0.0107423, 0.0093794, 0.0108423, -0.0012154, 0.0011279
3: -0.0049434, -0.0035703, -0.0049798, -0.0034669, -0.0012570, 0.0011665
4: -0.0001719, 0.0013146, -0.0002839, 0.0013540, -0.0012628, 0.0013608
5: 0.0028693, 0.0042760, 0.0028320, 0.0043820, -0.0012878, 0.0011951
6: -0.0109156, -0.0053344, -0.0110637, -0.0049138, -0.0051096, 0.0047417
7: 0.0047082, 0.0123094, 0.0041355, 0.0125111, -0.0064578, 0.0069588
8: 0.9925305, 0.9978849, 0.9921269, 0.9980270, -0.0045490, 0.0049019
9: -0.0139673, -0.0091069, -0.0140963, -0.0087407, -0.0044496, 0.0041293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020089, upper bound: 0.0020368
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019923, upper bound: 0.0020309
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0065681, 0.0088262, 0.0062168, 0.0089984, -0.0019168, 0.0020898
1: 0.0022712, 0.0025974, 0.0021610, 0.0026162, -0.0002696, 0.0003671
2: 0.0094801, 0.0107285, 0.0093775, 0.0109228, -0.0011554, 0.0010695
3: -0.0048757, -0.0035845, -0.0049498, -0.0033262, -0.0012579, 0.0010672
4: -0.0001565, 0.0012413, -0.0003797, 0.0013215, -0.0011553, 0.0012999
5: 0.0029387, 0.0042615, 0.0028449, 0.0044673, -0.0012242, 0.0011144
6: -0.0106406, -0.0053921, -0.0109608, -0.0045755, -0.0048573, 0.0043612
7: 0.0047869, 0.0119349, 0.0036087, 0.0123451, -0.0059078, 0.0066871
8: 0.9925859, 0.9976211, 0.9918025, 0.9979163, -0.0041692, 0.0046599
9: -0.0137278, -0.0091572, -0.0139902, -0.0084245, -0.0042534, 0.0037776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0022656
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0022683
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064450, 0.0088356, 0.0062168, 0.0089984, -0.0019472, 0.0019966
1: 0.0022534, 0.0025988, 0.0021610, 0.0026162, -0.0002725, 0.0003526
2: 0.0094749, 0.0107966, 0.0093775, 0.0109228, -0.0011039, 0.0010872
3: -0.0048811, -0.0035141, -0.0049498, -0.0033262, -0.0012024, 0.0010784
4: -0.0002327, 0.0012471, -0.0003797, 0.0013215, -0.0011674, 0.0012420
5: 0.0029332, 0.0043336, 0.0028449, 0.0044673, -0.0011696, 0.0011306
6: -0.0106624, -0.0051061, -0.0109608, -0.0045755, -0.0046406, 0.0044107
7: 0.0043973, 0.0119646, 0.0036087, 0.0123451, -0.0059696, 0.0063897
8: 0.9923114, 0.9976419, 0.9918025, 0.9979163, -0.0042139, 0.0044520
9: -0.0137468, -0.0089081, -0.0139902, -0.0084245, -0.0040639, 0.0038171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0022507
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020928, upper bound: 0.0022470
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065681, 0.0088262, 0.0061675, 0.0093072, -0.0023630, 0.0021776
1: 0.0022712, 0.0025974, 0.0018532, 0.0026293, -0.0002893, 0.0007039
2: 0.0094801, 0.0107285, 0.0091526, 0.0109500, -0.0012039, 0.0013917
3: -0.0048757, -0.0035845, -0.0050020, -0.0030221, -0.0016185, 0.0011451
4: -0.0001565, 0.0012413, -0.0004372, 0.0013780, -0.0012396, 0.0013848
5: 0.0029387, 0.0042615, 0.0027075, 0.0044961, -0.0012756, 0.0013093
6: -0.0106406, -0.0053921, -0.0112478, -0.0044610, -0.0050613, 0.0047647
7: 0.0047869, 0.0119349, 0.0031424, 0.0126341, -0.0063390, 0.0073170
8: 0.9925859, 0.9976211, 0.9916925, 0.9981436, -0.0044997, 0.0048556
9: -0.0137278, -0.0091572, -0.0141749, -0.0082238, -0.0045456, 0.0040533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0023297
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0023392
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064450, 0.0088356, 0.0061675, 0.0093072, -0.0024102, 0.0020892
1: 0.0022534, 0.0025988, 0.0018532, 0.0026293, -0.0002928, 0.0006904
2: 0.0094749, 0.0107966, 0.0091526, 0.0109500, -0.0011551, 0.0014242
3: -0.0048811, -0.0035141, -0.0050020, -0.0030221, -0.0015622, 0.0011591
4: -0.0002327, 0.0012471, -0.0004372, 0.0013780, -0.0012548, 0.0013295
5: 0.0029332, 0.0043336, 0.0027075, 0.0044961, -0.0012239, 0.0013330
6: -0.0106624, -0.0051061, -0.0112478, -0.0044610, -0.0048560, 0.0048359
7: 0.0043973, 0.0119646, 0.0031424, 0.0126341, -0.0064164, 0.0070319
8: 0.9923114, 0.9976419, 0.9916925, 0.9981436, -0.0045578, 0.0046586
9: -0.0137468, -0.0089081, -0.0141749, -0.0082238, -0.0043646, 0.0041028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0022962
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020926, upper bound: 0.0022994
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0065371, 0.0088472, 0.0061998, 0.0090350, -0.0019751, 0.0021685
1: 0.0022667, 0.0026005, 0.0021123, 0.0026166, -0.0002708, 0.0004291
2: 0.0094685, 0.0107457, 0.0093505, 0.0109322, -0.0011989, 0.0011113
3: -0.0048877, -0.0035668, -0.0049515, -0.0032740, -0.0013529, 0.0010717
4: -0.0001757, 0.0012543, -0.0003946, 0.0013233, -0.0011602, 0.0013535
5: 0.0029263, 0.0042796, 0.0028291, 0.0044772, -0.0012703, 0.0011400
6: -0.0106894, -0.0053200, -0.0109819, -0.0045360, -0.0050401, 0.0043994
7: 0.0046886, 0.0120014, 0.0035047, 0.0123543, -0.0059327, 0.0069934
8: 0.9925166, 0.9976679, 0.9917645, 0.9979273, -0.0041931, 0.0048353
9: -0.0137704, -0.0090944, -0.0139960, -0.0083737, -0.0044310, 0.0037935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021219, upper bound: 0.0022788
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021218, upper bound: 0.0022807
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064118, 0.0089004, 0.0061998, 0.0090350, -0.0020162, 0.0021369
1: 0.0021736, 0.0026016, 0.0021123, 0.0026166, -0.0003545, 0.0004145
2: 0.0094296, 0.0108149, 0.0093505, 0.0109322, -0.0011934, 0.0011359
3: -0.0048921, -0.0034215, -0.0049515, -0.0032740, -0.0012949, 0.0011651
4: -0.0002606, 0.0012590, -0.0003946, 0.0013233, -0.0011842, 0.0012926
5: 0.0029032, 0.0043530, 0.0028291, 0.0044772, -0.0012407, 0.0011620
6: -0.0107270, -0.0050288, -0.0109819, -0.0045360, -0.0048431, 0.0044678
7: 0.0042078, 0.0120254, 0.0035047, 0.0123543, -0.0061050, 0.0066783
8: 0.9922373, 0.9976914, 0.9917645, 0.9979273, -0.0042546, 0.0046270
9: -0.0137857, -0.0088135, -0.0139960, -0.0083737, -0.0042316, 0.0038759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021219, upper bound: 0.0022689
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021218, upper bound: 0.0022645
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065371, 0.0088472, 0.0061482, 0.0093571, -0.0024369, 0.0022582
1: 0.0022667, 0.0026005, 0.0018013, 0.0026298, -0.0002900, 0.0007686
2: 0.0094685, 0.0107457, 0.0091161, 0.0109607, -0.0012485, 0.0014515
3: -0.0048877, -0.0035668, -0.0050038, -0.0029650, -0.0017134, 0.0011479
4: -0.0001757, 0.0012543, -0.0004536, 0.0013799, -0.0012427, 0.0014395
5: 0.0029263, 0.0042796, 0.0026892, 0.0045074, -0.0013228, 0.0013370
6: -0.0106894, -0.0053200, -0.0112672, -0.0044162, -0.0052486, 0.0047990
7: 0.0046886, 0.0120014, 0.0030311, 0.0126436, -0.0063548, 0.0076268
8: 0.9925166, 0.9976679, 0.9916496, 0.9981542, -0.0045173, 0.0050353
9: -0.0137704, -0.0090944, -0.0141810, -0.0081681, -0.0047266, 0.0040634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021211, upper bound: 0.0023423
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021209, upper bound: 0.0023522
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064118, 0.0089004, 0.0061482, 0.0093571, -0.0025065, 0.0022337
1: 0.0021736, 0.0026016, 0.0018013, 0.0026298, -0.0003740, 0.0007549
2: 0.0094296, 0.0108149, 0.0091161, 0.0109607, -0.0012469, 0.0014933
3: -0.0048921, -0.0034215, -0.0050038, -0.0029650, -0.0016558, 0.0012423
4: -0.0002606, 0.0012590, -0.0004536, 0.0013799, -0.0012678, 0.0013830
5: 0.0029032, 0.0043530, 0.0026892, 0.0045074, -0.0012974, 0.0013725
6: -0.0107270, -0.0050288, -0.0112672, -0.0044162, -0.0050681, 0.0048759
7: 0.0042078, 0.0120254, 0.0030311, 0.0126436, -0.0065321, 0.0073358
8: 0.9922373, 0.9976914, 0.9916496, 0.9981542, -0.0045849, 0.0048429
9: -0.0137857, -0.0088135, -0.0141810, -0.0081681, -0.0045418, 0.0041490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021211, upper bound: 0.0023136
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021209, upper bound: 0.0023138
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0064294, 0.0090742, 0.0063524, 0.0089313, -0.0019902, 0.0023452
1: 0.0020481, 0.0026165, 0.0022400, 0.0026126, -0.0005055, 0.0003124
2: 0.0093169, 0.0108052, 0.0094220, 0.0108478, -0.0013362, 0.0011003
3: -0.0049510, -0.0033148, -0.0049358, -0.0034612, -0.0012364, 0.0013442
4: -0.0002613, 0.0013228, -0.0002901, 0.0013064, -0.0012522, 0.0013385
5: 0.0028140, 0.0043427, 0.0028771, 0.0043878, -0.0013406, 0.0011659
6: -0.0109958, -0.0050698, -0.0108849, -0.0048907, -0.0050895, 0.0046258
7: 0.0041304, 0.0123517, 0.0041040, 0.0122676, -0.0065331, 0.0068445
8: 0.9922765, 0.9979306, 0.9921048, 0.9978554, -0.0044378, 0.0048422
9: -0.0139943, -0.0088056, -0.0139406, -0.0087205, -0.0043765, 0.0041042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020941, upper bound: 0.0020245
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0020204
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064968, 0.0090959, 0.0063949, 0.0089300, -0.0019999, 0.0023140
1: 0.0021688, 0.0026293, 0.0022462, 0.0026124, -0.0003888, 0.0003236
2: 0.0093221, 0.0107679, 0.0094227, 0.0108243, -0.0012946, 0.0011057
3: -0.0050020, -0.0034540, -0.0049351, -0.0034855, -0.0012808, 0.0012388
4: -0.0002096, 0.0013780, -0.0002637, 0.0013055, -0.0012476, 0.0013866
5: 0.0027890, 0.0043032, 0.0028779, 0.0043629, -0.0013417, 0.0011715
6: -0.0111745, -0.0052265, -0.0108818, -0.0049896, -0.0052341, 0.0046483
7: 0.0044584, 0.0126339, 0.0042387, 0.0122633, -0.0064408, 0.0070906
8: 0.9924269, 0.9981201, 0.9921997, 0.9978524, -0.0044594, 0.0050036
9: -0.0141748, -0.0089793, -0.0139378, -0.0088067, -0.0045339, 0.0040840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020820, upper bound: 0.0020028
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0019982
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0064294, 0.0090742, 0.0063231, 0.0090097, -0.0019329, 0.0022801
1: 0.0020481, 0.0026165, 0.0022358, 0.0026239, -0.0004943, 0.0002980
2: 0.0093169, 0.0108052, 0.0093786, 0.0108640, -0.0013039, 0.0010687
3: -0.0049510, -0.0033148, -0.0049806, -0.0034444, -0.0011797, 0.0013061
4: -0.0002613, 0.0013228, -0.0003082, 0.0013549, -0.0012162, 0.0012771
5: 0.0028140, 0.0043427, 0.0028312, 0.0044050, -0.0012968, 0.0011323
6: -0.0109958, -0.0050698, -0.0110671, -0.0048227, -0.0048670, 0.0044926
7: 0.0041304, 0.0123517, 0.0040114, 0.0125157, -0.0063465, 0.0065305
8: 0.9922765, 0.9979306, 0.9920396, 0.9980302, -0.0043101, 0.0046229
9: -0.0139943, -0.0088056, -0.0140992, -0.0086613, -0.0041758, 0.0039860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020954, upper bound: 0.0020295
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020796, upper bound: 0.0020242
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064968, 0.0090959, 0.0063623, 0.0090083, -0.0019460, 0.0022247
1: 0.0021688, 0.0026293, 0.0022415, 0.0026237, -0.0003780, 0.0003090
2: 0.0093221, 0.0107679, 0.0093794, 0.0108423, -0.0012471, 0.0010759
3: -0.0050020, -0.0034540, -0.0049798, -0.0034669, -0.0012229, 0.0012067
4: -0.0002096, 0.0013780, -0.0002839, 0.0013540, -0.0012141, 0.0013239
5: 0.0027890, 0.0043032, 0.0028320, 0.0043820, -0.0012875, 0.0011399
6: -0.0111745, -0.0052265, -0.0110637, -0.0049138, -0.0050068, 0.0045229
7: 0.0044584, 0.0126339, 0.0041355, 0.0125111, -0.0062686, 0.0067701
8: 0.9924269, 0.9981201, 0.9921269, 0.9980270, -0.0043391, 0.0047807
9: -0.0141748, -0.0089793, -0.0140963, -0.0087407, -0.0043290, 0.0039743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020820, upper bound: 0.0020062
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0020000
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0065415, 0.0089011, 0.0062168, 0.0089984, -0.0020184, 0.0022272
1: 0.0022674, 0.0026083, 0.0021610, 0.0026162, -0.0002843, 0.0003869
2: 0.0094387, 0.0107432, 0.0093775, 0.0109228, -0.0012314, 0.0011257
3: -0.0049185, -0.0035693, -0.0049498, -0.0033262, -0.0013364, 0.0011253
4: -0.0001730, 0.0012877, -0.0003797, 0.0013215, -0.0012182, 0.0013849
5: 0.0028948, 0.0042770, 0.0028449, 0.0044673, -0.0013047, 0.0011739
6: -0.0108146, -0.0053304, -0.0109608, -0.0045755, -0.0051766, 0.0045972
7: 0.0047028, 0.0121719, 0.0036087, 0.0123451, -0.0062294, 0.0071220
8: 0.9925266, 0.9977881, 0.9918025, 0.9979163, -0.0043957, 0.0049662
9: -0.0138794, -0.0091034, -0.0139902, -0.0084245, -0.0045314, 0.0039832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0022597
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0022624
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063911, 0.0091184, 0.0062168, 0.0089984, -0.0020481, 0.0023994
1: 0.0019597, 0.0026122, 0.0021610, 0.0026162, -0.0005941, 0.0003724
2: 0.0092774, 0.0108264, 0.0093775, 0.0109228, -0.0013848, 0.0011430
3: -0.0049340, -0.0032132, -0.0049498, -0.0033262, -0.0012807, 0.0014227
4: -0.0002929, 0.0013044, -0.0003797, 0.0013215, -0.0012577, 0.0013268
5: 0.0028056, 0.0043652, 0.0028449, 0.0044673, -0.0013534, 0.0011897
6: -0.0109475, -0.0049806, -0.0109608, -0.0045755, -0.0050609, 0.0046453
7: 0.0039167, 0.0122577, 0.0036087, 0.0123451, -0.0066097, 0.0068234
8: 0.9921910, 0.9978695, 0.9918025, 0.9979163, -0.0044390, 0.0047891
9: -0.0139342, -0.0086982, -0.0139902, -0.0084245, -0.0043412, 0.0041258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0022506
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021072, upper bound: 0.0022468
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0065415, 0.0089011, 0.0061675, 0.0093072, -0.0023314, 0.0021279
1: 0.0022674, 0.0026083, 0.0018532, 0.0026293, -0.0002756, 0.0006914
2: 0.0094387, 0.0107432, 0.0091526, 0.0109500, -0.0011765, 0.0013805
3: -0.0049185, -0.0035693, -0.0050020, -0.0030221, -0.0015786, 0.0010909
4: -0.0001730, 0.0012877, -0.0004372, 0.0013780, -0.0011810, 0.0013513
5: 0.0028948, 0.0042770, 0.0027075, 0.0044961, -0.0012465, 0.0012838
6: -0.0108146, -0.0053304, -0.0112478, -0.0044610, -0.0049458, 0.0045682
7: 0.0047028, 0.0121719, 0.0031424, 0.0126341, -0.0060392, 0.0071311
8: 0.9925266, 0.9977881, 0.9916925, 0.9981436, -0.0042965, 0.0047448
9: -0.0138794, -0.0091034, -0.0141749, -0.0082238, -0.0044349, 0.0038617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021102, upper bound: 0.0022720
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021102, upper bound: 0.0022725
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063911, 0.0091184, 0.0061675, 0.0093072, -0.0023937, 0.0023663
1: 0.0019597, 0.0026122, 0.0018532, 0.0026293, -0.0005820, 0.0006781
2: 0.0092774, 0.0108264, 0.0091526, 0.0109500, -0.0013839, 0.0014261
3: -0.0049340, -0.0032132, -0.0050020, -0.0030221, -0.0015255, 0.0013852
4: -0.0002929, 0.0013044, -0.0004372, 0.0013780, -0.0012250, 0.0012974
5: 0.0028056, 0.0043652, 0.0027075, 0.0044961, -0.0013222, 0.0013142
6: -0.0109475, -0.0049806, -0.0112478, -0.0044610, -0.0048622, 0.0046584
7: 0.0039167, 0.0122577, 0.0031424, 0.0126341, -0.0064370, 0.0068510
8: 0.9921910, 0.9978695, 0.9916925, 0.9981436, -0.0043638, 0.0045887
9: -0.0139342, -0.0086982, -0.0141749, -0.0082238, -0.0042581, 0.0040183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.52 + 597.84 = 601.36 seconds
