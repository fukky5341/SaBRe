## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00045437


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041204, -0.0015831, -0.0041204, -0.0015831, -0.0025373, 0.0025373)
1: (0.0048627, 0.0066459, 0.0048627, 0.0066459, -0.0017832, 0.0017832)
2: (0.0103607, 0.0151389, 0.0103607, 0.0151389, -0.0042870, 0.0042870)
3: (-0.0048623, -0.0027278, -0.0048623, -0.0027278, -0.0021345, 0.0021345)
4: (0.0045089, 0.0052210, 0.0045089, 0.0052210, -0.0007070, 0.0007070)
5: (-0.0024429, -0.0008370, -0.0024429, -0.0008370, -0.0016058, 0.0016058)
6: (-0.0060710, -0.0053010, -0.0060710, -0.0053010, -0.0007700, 0.0007700)
7: (-0.0032708, -0.0017990, -0.0032708, -0.0017990, -0.0014718, 0.0014718)
8: (-0.0044902, -0.0013257, -0.0044902, -0.0013257, -0.0031645, 0.0031645)
9: (1.0004252, 1.0009804, 1.0004252, 1.0009804, -0.0005552, 0.0005552)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.55 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0005261, upper bound: 0.0005261

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005261, upper bound: 0.0005256
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 9, lower bound: -0.0005261, upper bound: 0.0005256
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041204, -0.0017020, -0.0041204, -0.0015831, -0.0025373, 0.0024184
1: 0.0049632, 0.0066459, 0.0048627, 0.0066459, -0.0016827, 0.0017832
2: 0.0103607, 0.0149270, 0.0103607, 0.0151389, -0.0042870, 0.0040746
3: -0.0047457, -0.0027278, -0.0048623, -0.0027278, -0.0020179, 0.0021345
4: 0.0045467, 0.0052210, 0.0045089, 0.0052210, -0.0006643, 0.0006983
5: -0.0023847, -0.0008370, -0.0024429, -0.0008370, -0.0015477, 0.0016058
6: -0.0060325, -0.0053010, -0.0060710, -0.0053010, -0.0007315, 0.0007700
7: -0.0032708, -0.0018938, -0.0032708, -0.0017990, -0.0014718, 0.0013770
8: -0.0043529, -0.0013257, -0.0044902, -0.0013257, -0.0030272, 0.0031645
9: 1.0004252, 1.0009799, 1.0004252, 1.0009804, -0.0005552, 0.0005547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
time: 0.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
time: 0.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0042032, -0.0017569, -0.0041204, -0.0016075, -0.0025958, 0.0023635
1: 0.0049989, 0.0067208, 0.0048822, 0.0066459, -0.0016470, 0.0018386
2: 0.0102126, 0.0148237, 0.0103607, 0.0150932, -0.0044232, 0.0040208
3: -0.0046996, -0.0026410, -0.0048390, -0.0027278, -0.0019718, 0.0021980
4: 0.0045626, 0.0052491, 0.0045165, 0.0052210, -0.0006584, 0.0007264
5: -0.0023471, -0.0008014, -0.0024286, -0.0008370, -0.0015101, 0.0016271
6: -0.0060157, -0.0052736, -0.0060633, -0.0053010, -0.0007147, 0.0007897
7: -0.0033426, -0.0019156, -0.0032708, -0.0018161, -0.0015265, 0.0013551
8: -0.0042855, -0.0012303, -0.0044605, -0.0013257, -0.0029598, 0.0032301
9: 1.0004148, 1.0010854, 1.0004252, 1.0009804, -0.0005655, 0.0006602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
time: 0.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 9, lower bound: -0.0005256, upper bound: 0.0005256

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041204, -0.0017020, -0.0041204, -0.0017020, -0.0024184, 0.0024184
1: 0.0049632, 0.0066459, 0.0049632, 0.0066459, -0.0016827, 0.0016827
2: 0.0103607, 0.0149270, 0.0103607, 0.0149270, -0.0040746, 0.0040746
3: -0.0047457, -0.0027278, -0.0047457, -0.0027278, -0.0020179, 0.0020179
4: 0.0045467, 0.0052210, 0.0045467, 0.0052210, -0.0006555, 0.0006555
5: -0.0023847, -0.0008370, -0.0023847, -0.0008370, -0.0015477, 0.0015477
6: -0.0060325, -0.0053010, -0.0060325, -0.0053010, -0.0007315, 0.0007315
7: -0.0032708, -0.0018938, -0.0032708, -0.0018938, -0.0013770, 0.0013770
8: -0.0043529, -0.0013257, -0.0043529, -0.0013257, -0.0030272, 0.0030272
9: 1.0004252, 1.0009799, 1.0004252, 1.0009799, -0.0005547, 0.0005547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004999, upper bound: 0.0004870
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005205, upper bound: 0.0005200
time: 0.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041204, -0.0017020, -0.0042032, -0.0017569, -0.0023635, 0.0025012
1: 0.0049632, 0.0066459, 0.0049989, 0.0067208, -0.0017576, 0.0016470
2: 0.0103607, 0.0149270, 0.0102126, 0.0148237, -0.0039927, 0.0042548
3: -0.0047457, -0.0027278, -0.0046996, -0.0026410, -0.0021046, 0.0019718
4: 0.0045467, 0.0052210, 0.0045626, 0.0052491, -0.0006909, 0.0006485
5: -0.0023847, -0.0008370, -0.0023471, -0.0008014, -0.0015833, 0.0015101
6: -0.0060325, -0.0053010, -0.0060157, -0.0052736, -0.0007589, 0.0007147
7: -0.0032708, -0.0018938, -0.0033426, -0.0019156, -0.0013551, 0.0014488
8: -0.0043529, -0.0013257, -0.0042855, -0.0012303, -0.0031226, 0.0029598
9: 1.0004252, 1.0009799, 1.0004148, 1.0010854, -0.0006602, 0.0005651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004999, upper bound: 0.0004870
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005205, upper bound: 0.0005200
time: 0.96 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042032, -0.0017569, -0.0041204, -0.0017020, -0.0025012, 0.0023635
1: 0.0049989, 0.0067208, 0.0049632, 0.0066459, -0.0016470, 0.0017576
2: 0.0102126, 0.0148237, 0.0103607, 0.0149270, -0.0042548, 0.0039927
3: -0.0046996, -0.0026410, -0.0047457, -0.0027278, -0.0019718, 0.0021046
4: 0.0045626, 0.0052491, 0.0045467, 0.0052210, -0.0006485, 0.0006909
5: -0.0023471, -0.0008014, -0.0023847, -0.0008370, -0.0015101, 0.0015833
6: -0.0060157, -0.0052736, -0.0060325, -0.0053010, -0.0007147, 0.0007589
7: -0.0033426, -0.0019156, -0.0032708, -0.0018938, -0.0014488, 0.0013551
8: -0.0042855, -0.0012303, -0.0043529, -0.0013257, -0.0029598, 0.0031226
9: 1.0004148, 1.0010854, 1.0004252, 1.0009799, -0.0005651, 0.0006602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004990, upper bound: 0.0004866
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005200, upper bound: 0.0005200
time: 0.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042032, -0.0017569, -0.0042032, -0.0017569, -0.0024463, 0.0024463
1: 0.0049989, 0.0067208, 0.0049989, 0.0067208, -0.0017219, 0.0017219
2: 0.0102126, 0.0148237, 0.0102126, 0.0148237, -0.0040912, 0.0040912
3: -0.0046996, -0.0026410, -0.0046996, -0.0026410, -0.0020586, 0.0020586
4: 0.0045626, 0.0052491, 0.0045626, 0.0052491, -0.0006842, 0.0006842
5: -0.0023471, -0.0008014, -0.0023471, -0.0008014, -0.0015457, 0.0015457
6: -0.0060157, -0.0052736, -0.0060157, -0.0052736, -0.0007421, 0.0007421
7: -0.0033426, -0.0019156, -0.0033426, -0.0019156, -0.0014269, 0.0014269
8: -0.0042855, -0.0012303, -0.0042855, -0.0012303, -0.0030552, 0.0030552
9: 1.0004148, 1.0010854, 1.0004148, 1.0010854, -0.0006706, 0.0006706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004990, upper bound: 0.0004866
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005200, upper bound: 0.0005200
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0004999, upper bound: 0.0004870
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0005205, upper bound: 0.0005200
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0004999, upper bound: 0.0004870
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0005205, upper bound: 0.0005200
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0004990, upper bound: 0.0004866
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0005200, upper bound: 0.0005200
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0004990, upper bound: 0.0004866
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 9, lower bound: -0.0005200, upper bound: 0.0005200

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039673, -0.0016609, -0.0040966, -0.0017023, -0.0022651, 0.0024357
1: 0.0049500, 0.0065324, 0.0049640, 0.0066283, -0.0016783, 0.0015684
2: 0.0106434, 0.0150106, 0.0104048, 0.0149265, -0.0037777, 0.0040760
3: -0.0047712, -0.0028611, -0.0047454, -0.0027485, -0.0020227, 0.0018843
4: 0.0045374, 0.0051767, 0.0045468, 0.0052142, -0.0006582, 0.0006126
5: -0.0024164, -0.0009253, -0.0023842, -0.0008506, -0.0015658, 0.0014588
6: -0.0060438, -0.0053486, -0.0060324, -0.0053083, -0.0007356, 0.0006837
7: -0.0031820, -0.0018924, -0.0032576, -0.0018953, -0.0012867, 0.0013652
8: -0.0044103, -0.0015114, -0.0043525, -0.0013547, -0.0030557, 0.0028411
9: 1.0004410, 1.0007906, 1.0004276, 1.0009505, -0.0005095, 0.0003630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004814, upper bound: 0.0004814
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004814, upper bound: 0.0004814
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040793, -0.0017028, -0.0041204, -0.0017020, -0.0023773, 0.0024176
1: 0.0049647, 0.0066208, 0.0049632, 0.0066459, -0.0016812, 0.0016576
2: 0.0104387, 0.0149255, 0.0103607, 0.0149270, -0.0039315, 0.0040725
3: -0.0047450, -0.0027597, -0.0047457, -0.0027278, -0.0020172, 0.0019860
4: 0.0045469, 0.0052100, 0.0045467, 0.0052210, -0.0006540, 0.0006355
5: -0.0023835, -0.0008651, -0.0023847, -0.0008370, -0.0015465, 0.0015196
6: -0.0060322, -0.0053135, -0.0060325, -0.0053010, -0.0007313, 0.0007190
7: -0.0032520, -0.0018965, -0.0032708, -0.0018938, -0.0013583, 0.0013743
8: -0.0043518, -0.0013784, -0.0043529, -0.0013257, -0.0030261, 0.0029745
9: 1.0004286, 1.0009283, 1.0004252, 1.0009799, -0.0005513, 0.0005031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004875, upper bound: 0.0004999
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004875, upper bound: 0.0005205
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039673, -0.0016609, -0.0041802, -0.0017572, -0.0022101, 0.0025193
1: 0.0049500, 0.0065324, 0.0049999, 0.0067044, -0.0017544, 0.0015325
2: 0.0106434, 0.0150106, 0.0102554, 0.0148231, -0.0036958, 0.0042593
3: -0.0047712, -0.0028611, -0.0046993, -0.0026608, -0.0021104, 0.0018382
4: 0.0045374, 0.0051767, 0.0045627, 0.0052425, -0.0006942, 0.0006056
5: -0.0024164, -0.0009253, -0.0023466, -0.0008150, -0.0016014, 0.0014213
6: -0.0060438, -0.0053486, -0.0060156, -0.0052806, -0.0007632, 0.0006670
7: -0.0031820, -0.0018924, -0.0033293, -0.0019172, -0.0012648, 0.0014368
8: -0.0044103, -0.0015114, -0.0042850, -0.0012585, -0.0031518, 0.0027736
9: 1.0004410, 1.0007906, 1.0004170, 1.0010567, -0.0006157, 0.0003736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004806, upper bound: 0.0004807
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004806, upper bound: 0.0004870
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040793, -0.0017028, -0.0042032, -0.0017569, -0.0023224, 0.0025005
1: 0.0049647, 0.0066208, 0.0049989, 0.0067208, -0.0017561, 0.0016219
2: 0.0104387, 0.0149255, 0.0102126, 0.0148237, -0.0038639, 0.0042527
3: -0.0047450, -0.0027597, -0.0046996, -0.0026410, -0.0021040, 0.0019399
4: 0.0045469, 0.0052100, 0.0045626, 0.0052491, -0.0006893, 0.0006317
5: -0.0023835, -0.0008651, -0.0023471, -0.0008014, -0.0015821, 0.0014820
6: -0.0060322, -0.0053135, -0.0060157, -0.0052736, -0.0007586, 0.0007022
7: -0.0032520, -0.0018965, -0.0033426, -0.0019156, -0.0013364, 0.0014461
8: -0.0043518, -0.0013784, -0.0042855, -0.0012303, -0.0031215, 0.0029071
9: 1.0004286, 1.0009283, 1.0004148, 1.0010854, -0.0006568, 0.0005134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004868, upper bound: 0.0004990
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004868, upper bound: 0.0004990
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040554, -0.0017100, -0.0040966, -0.0017023, -0.0023531, 0.0023866
1: 0.0049773, 0.0066135, 0.0049640, 0.0066283, -0.0016510, 0.0016495
2: 0.0104874, 0.0149153, 0.0104048, 0.0149265, -0.0039717, 0.0040192
3: -0.0047332, -0.0027712, -0.0047454, -0.0027485, -0.0019848, 0.0019742
4: 0.0045508, 0.0052057, 0.0045468, 0.0052142, -0.0006546, 0.0006504
5: -0.0023792, -0.0008868, -0.0023842, -0.0008506, -0.0015285, 0.0014974
6: -0.0060288, -0.0053198, -0.0060324, -0.0053083, -0.0007205, 0.0007126
7: -0.0032559, -0.0019120, -0.0032576, -0.0018953, -0.0013606, 0.0013456
8: -0.0043454, -0.0014115, -0.0043525, -0.0013547, -0.0029907, 0.0029411
9: 1.0004292, 1.0008996, 1.0004276, 1.0009505, -0.0005213, 0.0004719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004807, upper bound: 0.0004806
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004807, upper bound: 0.0004868
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041599, -0.0017578, -0.0041204, -0.0017020, -0.0024579, 0.0023626
1: 0.0050008, 0.0066928, 0.0049632, 0.0066459, -0.0016451, 0.0017296
2: 0.0102953, 0.0148219, 0.0103607, 0.0149270, -0.0041145, 0.0039903
3: -0.0046989, -0.0026759, -0.0047457, -0.0027278, -0.0019711, 0.0020698
4: 0.0045628, 0.0052374, 0.0045467, 0.0052210, -0.0006470, 0.0006735
5: -0.0023460, -0.0008295, -0.0023847, -0.0008370, -0.0015089, 0.0015552
6: -0.0060155, -0.0052863, -0.0060325, -0.0053010, -0.0007145, 0.0007462
7: -0.0033235, -0.0019185, -0.0032708, -0.0018938, -0.0014298, 0.0013523
8: -0.0042841, -0.0012849, -0.0043529, -0.0013257, -0.0029584, 0.0030681
9: 1.0004183, 1.0010327, 1.0004252, 1.0009799, -0.0005616, 0.0006075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004870, upper bound: 0.0004999
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004870, upper bound: 0.0004999
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040554, -0.0017100, -0.0041802, -0.0017572, -0.0022982, 0.0024701
1: 0.0049773, 0.0066135, 0.0049999, 0.0067044, -0.0017271, 0.0016136
2: 0.0104874, 0.0149153, 0.0102554, 0.0148231, -0.0037984, 0.0040968
3: -0.0047332, -0.0027712, -0.0046993, -0.0026608, -0.0020725, 0.0019282
4: 0.0045508, 0.0052057, 0.0045627, 0.0052425, -0.0006867, 0.0006409
5: -0.0023792, -0.0008868, -0.0023466, -0.0008150, -0.0015641, 0.0014598
6: -0.0060288, -0.0053198, -0.0060156, -0.0052806, -0.0007481, 0.0006959
7: -0.0032559, -0.0019120, -0.0033293, -0.0019172, -0.0013388, 0.0014172
8: -0.0043454, -0.0014115, -0.0042850, -0.0012585, -0.0030869, 0.0028736
9: 1.0004292, 1.0008996, 1.0004170, 1.0010567, -0.0006275, 0.0004826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004804, upper bound: 0.0004804
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004804, upper bound: 0.0004866
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041599, -0.0017578, -0.0042032, -0.0017569, -0.0024029, 0.0024454
1: 0.0050008, 0.0066928, 0.0049989, 0.0067208, -0.0017200, 0.0016939
2: 0.0102953, 0.0148219, 0.0102126, 0.0148237, -0.0039428, 0.0040888
3: -0.0046989, -0.0026759, -0.0046996, -0.0026410, -0.0020578, 0.0020238
4: 0.0045628, 0.0052374, 0.0045626, 0.0052491, -0.0006826, 0.0006588
5: -0.0023460, -0.0008295, -0.0023471, -0.0008014, -0.0015445, 0.0015177
6: -0.0060155, -0.0052863, -0.0060157, -0.0052736, -0.0007419, 0.0007294
7: -0.0033235, -0.0019185, -0.0033426, -0.0019156, -0.0014079, 0.0014241
8: -0.0042841, -0.0012849, -0.0042855, -0.0012303, -0.0030538, 0.0030006
9: 1.0004183, 1.0010327, 1.0004148, 1.0010854, -0.0006671, 0.0006179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004990
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0005200
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004814, upper bound: 0.0004814
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004814, upper bound: 0.0004814
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004875, upper bound: 0.0004999
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004875, upper bound: 0.0005205
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004806, upper bound: 0.0004807
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004806, upper bound: 0.0004870
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004868, upper bound: 0.0004990
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004868, upper bound: 0.0004990
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004807, upper bound: 0.0004806
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004807, upper bound: 0.0004868
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004870, upper bound: 0.0004999
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004870, upper bound: 0.0004999
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004804, upper bound: 0.0004804
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004804, upper bound: 0.0004866
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0004990
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 9, lower bound: -0.0004866, upper bound: 0.0005200

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039673, -0.0016609, -0.0039673, -0.0016609, -0.0023065, 0.0023065
1: 0.0049500, 0.0065324, 0.0049500, 0.0065324, -0.0015824, 0.0015824
2: 0.0106434, 0.0150106, 0.0106434, 0.0150106, -0.0038269, 0.0038269
3: -0.0047712, -0.0028611, -0.0047712, -0.0028611, -0.0019101, 0.0019101
4: 0.0045374, 0.0051767, 0.0045374, 0.0051767, -0.0006228, 0.0006228
5: -0.0024164, -0.0009253, -0.0024164, -0.0009253, -0.0014911, 0.0014911
6: -0.0060438, -0.0053486, -0.0060438, -0.0053486, -0.0006952, 0.0006952
7: -0.0031820, -0.0018924, -0.0031820, -0.0018924, -0.0012895, 0.0012895
8: -0.0044103, -0.0015114, -0.0044103, -0.0015114, -0.0028989, 0.0028989
9: 1.0004410, 1.0007906, 1.0004410, 1.0007906, -0.0003496, 0.0003496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004726, upper bound: 0.0004736
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039673, -0.0016609, -0.0040793, -0.0017028, -0.0022646, 0.0024184
1: 0.0049500, 0.0065324, 0.0049647, 0.0066208, -0.0016708, 0.0015677
2: 0.0106434, 0.0150106, 0.0104387, 0.0149255, -0.0037765, 0.0040534
3: -0.0047712, -0.0028611, -0.0047450, -0.0027597, -0.0020115, 0.0018839
4: 0.0045374, 0.0051767, 0.0045469, 0.0052100, -0.0006538, 0.0006113
5: -0.0024164, -0.0009253, -0.0023835, -0.0008651, -0.0015512, 0.0014582
6: -0.0060438, -0.0053486, -0.0060322, -0.0053135, -0.0007304, 0.0006836
7: -0.0031820, -0.0018924, -0.0032520, -0.0018965, -0.0012855, 0.0013596
8: -0.0044103, -0.0015114, -0.0043518, -0.0013784, -0.0030319, 0.0028404
9: 1.0004410, 1.0007906, 1.0004286, 1.0009283, -0.0004873, 0.0003620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004726, upper bound: 0.0004736
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004780
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040793, -0.0017028, -0.0039673, -0.0016609, -0.0024184, 0.0022646
1: 0.0049647, 0.0066208, 0.0049500, 0.0065324, -0.0015677, 0.0016708
2: 0.0104387, 0.0149255, 0.0106434, 0.0150106, -0.0040535, 0.0037765
3: -0.0047450, -0.0027597, -0.0047712, -0.0028611, -0.0018839, 0.0020115
4: 0.0045469, 0.0052100, 0.0045374, 0.0051767, -0.0006113, 0.0006538
5: -0.0023835, -0.0008651, -0.0024164, -0.0009253, -0.0014582, 0.0015512
6: -0.0060322, -0.0053135, -0.0060438, -0.0053486, -0.0006836, 0.0007304
7: -0.0032520, -0.0018965, -0.0031820, -0.0018924, -0.0013596, 0.0012855
8: -0.0043518, -0.0013784, -0.0044103, -0.0015114, -0.0028404, 0.0030319
9: 1.0004286, 1.0009283, 1.0004410, 1.0007906, -0.0003620, 0.0004873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004791, upper bound: 0.0004907
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040793, -0.0017028, -0.0040793, -0.0017028, -0.0023765, 0.0023765
1: 0.0049647, 0.0066208, 0.0049647, 0.0066208, -0.0016561, 0.0016561
2: 0.0104387, 0.0149255, 0.0104387, 0.0149255, -0.0039294, 0.0039294
3: -0.0047450, -0.0027597, -0.0047450, -0.0027597, -0.0019853, 0.0019853
4: 0.0045469, 0.0052100, 0.0045469, 0.0052100, -0.0006343, 0.0006343
5: -0.0023835, -0.0008651, -0.0023835, -0.0008651, -0.0015184, 0.0015184
6: -0.0060322, -0.0053135, -0.0060322, -0.0053135, -0.0007188, 0.0007188
7: -0.0032520, -0.0018965, -0.0032520, -0.0018965, -0.0013555, 0.0013555
8: -0.0043518, -0.0013784, -0.0043518, -0.0013784, -0.0029734, 0.0029734
9: 1.0004286, 1.0009283, 1.0004286, 1.0009283, -0.0004997, 0.0004997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004791, upper bound: 0.0004907
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0005112
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039673, -0.0016609, -0.0040554, -0.0017100, -0.0022573, 0.0023945
1: 0.0049500, 0.0065324, 0.0049773, 0.0066135, -0.0016635, 0.0015551
2: 0.0106434, 0.0150106, 0.0104874, 0.0149153, -0.0037701, 0.0040209
3: -0.0047712, -0.0028611, -0.0047332, -0.0027712, -0.0020000, 0.0018721
4: 0.0045374, 0.0051767, 0.0045508, 0.0052057, -0.0006606, 0.0006193
5: -0.0024164, -0.0009253, -0.0023792, -0.0008868, -0.0015296, 0.0014539
6: -0.0060438, -0.0053486, -0.0060288, -0.0053198, -0.0007241, 0.0006801
7: -0.0031820, -0.0018924, -0.0032559, -0.0019120, -0.0012699, 0.0013635
8: -0.0044103, -0.0015114, -0.0043454, -0.0014115, -0.0029989, 0.0028340
9: 1.0004410, 1.0007906, 1.0004292, 1.0008996, -0.0004586, 0.0003614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004716, upper bound: 0.0004727
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039673, -0.0016609, -0.0041599, -0.0017578, -0.0022095, 0.0024990
1: 0.0049500, 0.0065324, 0.0050008, 0.0066928, -0.0017429, 0.0015315
2: 0.0106434, 0.0150106, 0.0102953, 0.0148219, -0.0036943, 0.0042264
3: -0.0047712, -0.0028611, -0.0046989, -0.0026759, -0.0020953, 0.0018377
4: 0.0045374, 0.0051767, 0.0045628, 0.0052374, -0.0006869, 0.0006044
5: -0.0024164, -0.0009253, -0.0023460, -0.0008295, -0.0015869, 0.0014206
6: -0.0060438, -0.0053486, -0.0060155, -0.0052863, -0.0007576, 0.0006668
7: -0.0031820, -0.0018924, -0.0033235, -0.0019185, -0.0012635, 0.0014311
8: -0.0044103, -0.0015114, -0.0042841, -0.0012849, -0.0031254, 0.0027727
9: 1.0004410, 1.0007906, 1.0004183, 1.0010327, -0.0005918, 0.0003723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004716, upper bound: 0.0004799
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004773
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040793, -0.0017028, -0.0040554, -0.0017100, -0.0023693, 0.0023526
1: 0.0049647, 0.0066208, 0.0049773, 0.0066135, -0.0016488, 0.0016435
2: 0.0104387, 0.0149255, 0.0104874, 0.0149153, -0.0039966, 0.0039705
3: -0.0047450, -0.0027597, -0.0047332, -0.0027712, -0.0019738, 0.0019735
4: 0.0045469, 0.0052100, 0.0045508, 0.0052057, -0.0006491, 0.0006503
5: -0.0023835, -0.0008651, -0.0023792, -0.0008868, -0.0014967, 0.0015140
6: -0.0060322, -0.0053135, -0.0060288, -0.0053198, -0.0007125, 0.0007153
7: -0.0032520, -0.0018965, -0.0032559, -0.0019120, -0.0013400, 0.0013595
8: -0.0043518, -0.0013784, -0.0043454, -0.0014115, -0.0029403, 0.0029670
9: 1.0004286, 1.0009283, 1.0004292, 1.0008996, -0.0004710, 0.0004991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004777, upper bound: 0.0004885
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040793, -0.0017028, -0.0041599, -0.0017578, -0.0023215, 0.0024571
1: 0.0049647, 0.0066208, 0.0050008, 0.0066928, -0.0017281, 0.0016200
2: 0.0104387, 0.0149255, 0.0102953, 0.0148219, -0.0038615, 0.0041124
3: -0.0047450, -0.0027597, -0.0046989, -0.0026759, -0.0020691, 0.0019392
4: 0.0045469, 0.0052100, 0.0045628, 0.0052374, -0.0006723, 0.0006305
5: -0.0023835, -0.0008651, -0.0023460, -0.0008295, -0.0015540, 0.0014808
6: -0.0060322, -0.0053135, -0.0060155, -0.0052863, -0.0007460, 0.0007020
7: -0.0032520, -0.0018965, -0.0033235, -0.0019185, -0.0013335, 0.0014270
8: -0.0043518, -0.0013784, -0.0042841, -0.0012849, -0.0030669, 0.0029057
9: 1.0004286, 1.0009283, 1.0004183, 1.0010327, -0.0006042, 0.0005100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004777, upper bound: 0.0005128
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0005105
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040554, -0.0017100, -0.0039673, -0.0016609, -0.0023945, 0.0022573
1: 0.0049773, 0.0066135, 0.0049500, 0.0065324, -0.0015551, 0.0016635
2: 0.0104874, 0.0149153, 0.0106434, 0.0150106, -0.0040209, 0.0037701
3: -0.0047332, -0.0027712, -0.0047712, -0.0028611, -0.0018721, 0.0020000
4: 0.0045508, 0.0052057, 0.0045374, 0.0051767, -0.0006193, 0.0006606
5: -0.0023792, -0.0008868, -0.0024164, -0.0009253, -0.0014539, 0.0015296
6: -0.0060288, -0.0053198, -0.0060438, -0.0053486, -0.0006801, 0.0007241
7: -0.0032559, -0.0019120, -0.0031820, -0.0018924, -0.0013635, 0.0012699
8: -0.0043454, -0.0014115, -0.0044103, -0.0015114, -0.0028340, 0.0029989
9: 1.0004292, 1.0008996, 1.0004410, 1.0007906, -0.0003614, 0.0004586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004718, upper bound: 0.0004730
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040554, -0.0017100, -0.0040793, -0.0017028, -0.0023526, 0.0023693
1: 0.0049773, 0.0066135, 0.0049647, 0.0066208, -0.0016435, 0.0016488
2: 0.0104874, 0.0149153, 0.0104387, 0.0149255, -0.0039705, 0.0039966
3: -0.0047332, -0.0027712, -0.0047450, -0.0027597, -0.0019735, 0.0019738
4: 0.0045508, 0.0052057, 0.0045469, 0.0052100, -0.0006503, 0.0006491
5: -0.0023792, -0.0008868, -0.0023835, -0.0008651, -0.0015140, 0.0014967
6: -0.0060288, -0.0053198, -0.0060322, -0.0053135, -0.0007153, 0.0007125
7: -0.0032559, -0.0019120, -0.0032520, -0.0018965, -0.0013595, 0.0013400
8: -0.0043454, -0.0014115, -0.0043518, -0.0013784, -0.0029670, 0.0029403
9: 1.0004292, 1.0008996, 1.0004286, 1.0009283, -0.0004991, 0.0004710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004718, upper bound: 0.0004797
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004766
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041599, -0.0017578, -0.0039673, -0.0016609, -0.0024990, 0.0022095
1: 0.0050008, 0.0066928, 0.0049500, 0.0065324, -0.0015315, 0.0017429
2: 0.0102953, 0.0148219, 0.0106434, 0.0150106, -0.0042264, 0.0036943
3: -0.0046989, -0.0026759, -0.0047712, -0.0028611, -0.0018377, 0.0020953
4: 0.0045628, 0.0052374, 0.0045374, 0.0051767, -0.0006044, 0.0006869
5: -0.0023460, -0.0008295, -0.0024164, -0.0009253, -0.0014206, 0.0015869
6: -0.0060155, -0.0052863, -0.0060438, -0.0053486, -0.0006668, 0.0007576
7: -0.0033235, -0.0019185, -0.0031820, -0.0018924, -0.0014311, 0.0012635
8: -0.0042841, -0.0012849, -0.0044103, -0.0015114, -0.0027727, 0.0031254
9: 1.0004183, 1.0010327, 1.0004410, 1.0007906, -0.0003723, 0.0005918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004785, upper bound: 0.0004907
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041599, -0.0017578, -0.0040793, -0.0017028, -0.0024571, 0.0023215
1: 0.0050008, 0.0066928, 0.0049647, 0.0066208, -0.0016200, 0.0017281
2: 0.0102953, 0.0148219, 0.0104387, 0.0149255, -0.0041124, 0.0038615
3: -0.0046989, -0.0026759, -0.0047450, -0.0027597, -0.0019392, 0.0020691
4: 0.0045628, 0.0052374, 0.0045469, 0.0052100, -0.0006305, 0.0006723
5: -0.0023460, -0.0008295, -0.0023835, -0.0008651, -0.0014808, 0.0015540
6: -0.0060155, -0.0052863, -0.0060322, -0.0053135, -0.0007020, 0.0007460
7: -0.0033235, -0.0019185, -0.0032520, -0.0018965, -0.0014270, 0.0013335
8: -0.0042841, -0.0012849, -0.0043518, -0.0013784, -0.0029057, 0.0030669
9: 1.0004183, 1.0010327, 1.0004286, 1.0009283, -0.0005100, 0.0006042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004785, upper bound: 0.0004730
time: 3.30 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0005112
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040554, -0.0017100, -0.0040554, -0.0017100, -0.0023454, 0.0023454
1: 0.0049773, 0.0066135, 0.0049773, 0.0066135, -0.0016362, 0.0016362
2: 0.0104874, 0.0149153, 0.0104874, 0.0149153, -0.0038511, 0.0038511
3: -0.0047332, -0.0027712, -0.0047332, -0.0027712, -0.0019620, 0.0019620
4: 0.0045508, 0.0052057, 0.0045508, 0.0052057, -0.0006507, 0.0006507
5: -0.0023792, -0.0008868, -0.0023792, -0.0008868, -0.0014924, 0.0014924
6: -0.0060288, -0.0053198, -0.0060288, -0.0053198, -0.0007090, 0.0007090
7: -0.0032559, -0.0019120, -0.0032559, -0.0019120, -0.0013439, 0.0013439
8: -0.0043454, -0.0014115, -0.0043454, -0.0014115, -0.0029339, 0.0029339
9: 1.0004292, 1.0008996, 1.0004292, 1.0008996, -0.0004704, 0.0004704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004712, upper bound: 0.0004724
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040554, -0.0017100, -0.0041599, -0.0017578, -0.0022976, 0.0024498
1: 0.0049773, 0.0066135, 0.0050008, 0.0066928, -0.0017156, 0.0016127
2: 0.0104874, 0.0149153, 0.0102953, 0.0148219, -0.0037969, 0.0040717
3: -0.0047332, -0.0027712, -0.0046989, -0.0026759, -0.0020574, 0.0019277
4: 0.0045508, 0.0052057, 0.0045628, 0.0052374, -0.0006818, 0.0006396
5: -0.0023792, -0.0008868, -0.0023460, -0.0008295, -0.0015497, 0.0014592
6: -0.0060288, -0.0053198, -0.0060155, -0.0052863, -0.0007425, 0.0006957
7: -0.0032559, -0.0019120, -0.0033235, -0.0019185, -0.0013375, 0.0014115
8: -0.0043454, -0.0014115, -0.0042841, -0.0012849, -0.0030605, 0.0028727
9: 1.0004292, 1.0008996, 1.0004183, 1.0010327, -0.0006036, 0.0004812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004712, upper bound: 0.0004724
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004763
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041599, -0.0017578, -0.0040554, -0.0017100, -0.0024498, 0.0022976
1: 0.0050008, 0.0066928, 0.0049773, 0.0066135, -0.0016127, 0.0017156
2: 0.0102953, 0.0148219, 0.0104874, 0.0149153, -0.0040717, 0.0037969
3: -0.0046989, -0.0026759, -0.0047332, -0.0027712, -0.0019277, 0.0020574
4: 0.0045628, 0.0052374, 0.0045508, 0.0052057, -0.0006396, 0.0006818
5: -0.0023460, -0.0008295, -0.0023792, -0.0008868, -0.0014592, 0.0015497
6: -0.0060155, -0.0052863, -0.0060288, -0.0053198, -0.0006957, 0.0007425
7: -0.0033235, -0.0019185, -0.0032559, -0.0019120, -0.0014115, 0.0013375
8: -0.0042841, -0.0012849, -0.0043454, -0.0014115, -0.0028727, 0.0030605
9: 1.0004183, 1.0010327, 1.0004292, 1.0008996, -0.0004812, 0.0006036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004775, upper bound: 0.0004885
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041599, -0.0017578, -0.0041599, -0.0017578, -0.0024020, 0.0024020
1: 0.0050008, 0.0066928, 0.0050008, 0.0066928, -0.0016920, 0.0016920
2: 0.0102953, 0.0148219, 0.0102953, 0.0148219, -0.0039407, 0.0039407
3: -0.0046989, -0.0026759, -0.0046989, -0.0026759, -0.0020230, 0.0020230
4: 0.0045628, 0.0052374, 0.0045628, 0.0052374, -0.0006576, 0.0006576
5: -0.0023460, -0.0008295, -0.0023460, -0.0008295, -0.0015165, 0.0015165
6: -0.0060155, -0.0052863, -0.0060155, -0.0052863, -0.0007292, 0.0007292
7: -0.0033235, -0.0019185, -0.0033235, -0.0019185, -0.0014050, 0.0014050
8: -0.0042841, -0.0012849, -0.0042841, -0.0012849, -0.0029993, 0.0029993
9: 1.0004183, 1.0010327, 1.0004183, 1.0010327, -0.0006144, 0.0006144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004775, upper bound: 0.0005128
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004726, upper bound: 0.0004736
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004726, upper bound: 0.0004736
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004780
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004791, upper bound: 0.0004907
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004791, upper bound: 0.0004907
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0005112
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004716, upper bound: 0.0004727
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004716, upper bound: 0.0004799
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004773
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004777, upper bound: 0.0004885
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004777, upper bound: 0.0005128
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0005105
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004718, upper bound: 0.0004730
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004718, upper bound: 0.0004797
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004766
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004785, upper bound: 0.0004907
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004785, upper bound: 0.0004730
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0005112
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004712, upper bound: 0.0004724
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004712, upper bound: 0.0004724
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004763
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004775, upper bound: 0.0004885
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004775, upper bound: 0.0005128
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0039673, -0.0016609, -0.0022903, 0.0023000
1: 0.0049610, 0.0065324, 0.0049500, 0.0065324, -0.0015713, 0.0015824
2: 0.0106815, 0.0149987, 0.0106434, 0.0150106, -0.0037878, 0.0038179
3: -0.0047633, -0.0028611, -0.0047712, -0.0028611, -0.0019021, 0.0019101
4: 0.0045394, 0.0051765, 0.0045374, 0.0051767, -0.0006196, 0.0006187
5: -0.0024131, -0.0009460, -0.0024164, -0.0009253, -0.0014878, 0.0014704
6: -0.0060418, -0.0053514, -0.0060438, -0.0053486, -0.0006931, 0.0006925
7: -0.0031820, -0.0019151, -0.0031820, -0.0018924, -0.0012895, 0.0012669
8: -0.0044025, -0.0015385, -0.0044103, -0.0015114, -0.0028911, 0.0028719
9: 1.0004410, 1.0007743, 1.0004410, 1.0007906, -0.0003496, 0.0003333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0039638, -0.0016624, -0.0022804, 0.0023272
1: 0.0049754, 0.0065367, 0.0049544, 0.0065324, -0.0015570, 0.0015823
2: 0.0107018, 0.0150746, 0.0106519, 0.0150077, -0.0037830, 0.0038899
3: -0.0047573, -0.0028598, -0.0047687, -0.0028611, -0.0018961, 0.0019089
4: 0.0045393, 0.0051763, 0.0045379, 0.0051767, -0.0006264, 0.0006172
5: -0.0024557, -0.0009568, -0.0024156, -0.0009299, -0.0015259, 0.0014588
6: -0.0060468, -0.0053530, -0.0060433, -0.0053493, -0.0006975, 0.0006904
7: -0.0032045, -0.0019478, -0.0031820, -0.0019015, -0.0013030, 0.0012341
8: -0.0044575, -0.0015528, -0.0044085, -0.0015175, -0.0029400, 0.0028556
9: 1.0004396, 1.0007663, 1.0004410, 1.0007870, -0.0003474, 0.0003253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0040793, -0.0017028, -0.0022484, 0.0024119
1: 0.0049610, 0.0065324, 0.0049647, 0.0066208, -0.0016597, 0.0015677
2: 0.0106815, 0.0149987, 0.0104387, 0.0149255, -0.0037373, 0.0040444
3: -0.0047633, -0.0028611, -0.0047450, -0.0027597, -0.0020036, 0.0018839
4: 0.0045394, 0.0051765, 0.0045469, 0.0052100, -0.0006506, 0.0006073
5: -0.0024131, -0.0009460, -0.0023835, -0.0008651, -0.0015479, 0.0014375
6: -0.0060418, -0.0053514, -0.0060322, -0.0053135, -0.0007283, 0.0006809
7: -0.0031820, -0.0019151, -0.0032520, -0.0018965, -0.0012855, 0.0013369
8: -0.0044025, -0.0015385, -0.0043518, -0.0013784, -0.0030241, 0.0028133
9: 1.0004410, 1.0007743, 1.0004286, 1.0009283, -0.0004873, 0.0003457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0040759, -0.0017044, -0.0022384, 0.0024392
1: 0.0049754, 0.0065367, 0.0049692, 0.0066208, -0.0016454, 0.0015674
2: 0.0107018, 0.0150746, 0.0104466, 0.0149226, -0.0037323, 0.0041159
3: -0.0047573, -0.0028598, -0.0047424, -0.0027597, -0.0019976, 0.0018826
4: 0.0045393, 0.0051763, 0.0045474, 0.0052100, -0.0006574, 0.0006058
5: -0.0024557, -0.0009568, -0.0023827, -0.0008698, -0.0015859, 0.0014258
6: -0.0060468, -0.0053530, -0.0060317, -0.0053141, -0.0007327, 0.0006788
7: -0.0032045, -0.0019478, -0.0032520, -0.0019049, -0.0012996, 0.0013042
8: -0.0044575, -0.0015528, -0.0043499, -0.0013841, -0.0030733, 0.0027971
9: 1.0004396, 1.0007663, 1.0004286, 1.0009246, -0.0004849, 0.0003377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0039673, -0.0016609, -0.0024021, 0.0022580
1: 0.0049768, 0.0066208, 0.0049500, 0.0065324, -0.0015556, 0.0016708
2: 0.0104779, 0.0149136, 0.0106434, 0.0150106, -0.0040157, 0.0037666
3: -0.0047365, -0.0027597, -0.0047712, -0.0028611, -0.0018754, 0.0020115
4: 0.0045490, 0.0052098, 0.0045374, 0.0051767, -0.0006077, 0.0006498
5: -0.0023801, -0.0008863, -0.0024164, -0.0009253, -0.0014547, 0.0015300
6: -0.0060301, -0.0053163, -0.0060438, -0.0053486, -0.0006815, 0.0007276
7: -0.0032520, -0.0019201, -0.0031820, -0.0018924, -0.0013596, 0.0012619
8: -0.0043444, -0.0014065, -0.0044103, -0.0015114, -0.0028330, 0.0030038
9: 1.0004286, 1.0009116, 1.0004410, 1.0007906, -0.0003620, 0.0004706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0039638, -0.0016624, -0.0023931, 0.0022877
1: 0.0049888, 0.0066256, 0.0049544, 0.0065324, -0.0015436, 0.0016712
2: 0.0104947, 0.0149935, 0.0106519, 0.0150077, -0.0040080, 0.0038376
3: -0.0047309, -0.0027582, -0.0047687, -0.0028611, -0.0018698, 0.0020105
4: 0.0045490, 0.0052096, 0.0045379, 0.0051767, -0.0006140, 0.0006483
5: -0.0024220, -0.0008967, -0.0024156, -0.0009299, -0.0014921, 0.0015189
6: -0.0060353, -0.0053176, -0.0060433, -0.0053493, -0.0006860, 0.0007257
7: -0.0032766, -0.0019509, -0.0031820, -0.0019015, -0.0013752, 0.0012311
8: -0.0044012, -0.0014181, -0.0044085, -0.0015175, -0.0028838, 0.0029904
9: 1.0004270, 1.0009035, 1.0004410, 1.0007870, -0.0003600, 0.0004625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0040793, -0.0017028, -0.0023602, 0.0023700
1: 0.0049768, 0.0066208, 0.0049647, 0.0066208, -0.0016440, 0.0016561
2: 0.0104779, 0.0149136, 0.0104387, 0.0149255, -0.0038900, 0.0039199
3: -0.0047365, -0.0027597, -0.0047450, -0.0027597, -0.0019768, 0.0019853
4: 0.0045490, 0.0052098, 0.0045469, 0.0052100, -0.0006311, 0.0006302
5: -0.0023801, -0.0008863, -0.0023835, -0.0008651, -0.0015149, 0.0014972
6: -0.0060301, -0.0053163, -0.0060322, -0.0053135, -0.0007167, 0.0007160
7: -0.0032520, -0.0019201, -0.0032520, -0.0018965, -0.0013555, 0.0013319
8: -0.0043444, -0.0014065, -0.0043518, -0.0013784, -0.0029660, 0.0029453
9: 1.0004286, 1.0009116, 1.0004286, 1.0009283, -0.0004997, 0.0004830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0040759, -0.0017044, -0.0023512, 0.0023997
1: 0.0049888, 0.0066256, 0.0049692, 0.0066208, -0.0016320, 0.0016564
2: 0.0104947, 0.0149935, 0.0104466, 0.0149226, -0.0038852, 0.0039902
3: -0.0047309, -0.0027582, -0.0047424, -0.0027597, -0.0019712, 0.0019842
4: 0.0045490, 0.0052096, 0.0045474, 0.0052100, -0.0006377, 0.0006288
5: -0.0024220, -0.0008967, -0.0023827, -0.0008698, -0.0015522, 0.0014860
6: -0.0060353, -0.0053176, -0.0060317, -0.0053141, -0.0007212, 0.0007141
7: -0.0032766, -0.0019509, -0.0032520, -0.0019049, -0.0013717, 0.0013011
8: -0.0044012, -0.0014181, -0.0043499, -0.0013841, -0.0030171, 0.0029318
9: 1.0004270, 1.0009035, 1.0004286, 1.0009246, -0.0004976, 0.0004749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0040554, -0.0017100, -0.0022411, 0.0023880
1: 0.0049610, 0.0065324, 0.0049773, 0.0066135, -0.0016525, 0.0015551
2: 0.0106815, 0.0149987, 0.0104874, 0.0149153, -0.0037310, 0.0040118
3: -0.0047633, -0.0028611, -0.0047332, -0.0027712, -0.0019921, 0.0018721
4: 0.0045394, 0.0051765, 0.0045508, 0.0052057, -0.0006573, 0.0006152
5: -0.0024131, -0.0009460, -0.0023792, -0.0008868, -0.0015263, 0.0014332
6: -0.0060418, -0.0053514, -0.0060288, -0.0053198, -0.0007220, 0.0006774
7: -0.0031820, -0.0019151, -0.0032559, -0.0019120, -0.0012699, 0.0013408
8: -0.0044025, -0.0015385, -0.0043454, -0.0014115, -0.0029911, 0.0028069
9: 1.0004410, 1.0007743, 1.0004292, 1.0008996, -0.0004586, 0.0003451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0040517, -0.0017120, -0.0022308, 0.0024150
1: 0.0049754, 0.0065367, 0.0049818, 0.0066135, -0.0016381, 0.0015548
2: 0.0107018, 0.0150746, 0.0104961, 0.0149118, -0.0037254, 0.0040834
3: -0.0047573, -0.0028598, -0.0047303, -0.0027712, -0.0019861, 0.0018705
4: 0.0045393, 0.0051763, 0.0045514, 0.0052056, -0.0006642, 0.0006135
5: -0.0024557, -0.0009568, -0.0023782, -0.0008918, -0.0015639, 0.0014214
6: -0.0060468, -0.0053530, -0.0060282, -0.0053204, -0.0007264, 0.0006752
7: -0.0032045, -0.0019478, -0.0032559, -0.0019205, -0.0012839, 0.0013081
8: -0.0044575, -0.0015528, -0.0043431, -0.0014176, -0.0030398, 0.0027903
9: 1.0004396, 1.0007663, 1.0004292, 1.0008957, -0.0004561, 0.0003371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0041599, -0.0017578, -0.0021933, 0.0024925
1: 0.0049610, 0.0065324, 0.0050008, 0.0066928, -0.0017318, 0.0015315
2: 0.0106815, 0.0149987, 0.0102953, 0.0148219, -0.0036552, 0.0042173
3: -0.0047633, -0.0028611, -0.0046989, -0.0026759, -0.0020874, 0.0018377
4: 0.0045394, 0.0051765, 0.0045628, 0.0052374, -0.0006837, 0.0006003
5: -0.0024131, -0.0009460, -0.0023460, -0.0008295, -0.0015836, 0.0014000
6: -0.0060418, -0.0053514, -0.0060155, -0.0052863, -0.0007555, 0.0006641
7: -0.0031820, -0.0019151, -0.0033235, -0.0019185, -0.0012635, 0.0014084
8: -0.0044025, -0.0015385, -0.0042841, -0.0012849, -0.0031176, 0.0027457
9: 1.0004410, 1.0007743, 1.0004183, 1.0010327, -0.0005918, 0.0003560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0041561, -0.0017597, -0.0021831, 0.0025194
1: 0.0049754, 0.0065367, 0.0050054, 0.0066928, -0.0017175, 0.0015313
2: 0.0107018, 0.0150746, 0.0103044, 0.0148185, -0.0036496, 0.0042879
3: -0.0047573, -0.0028598, -0.0046959, -0.0026759, -0.0020814, 0.0018362
4: 0.0045393, 0.0051763, 0.0045633, 0.0052374, -0.0006906, 0.0005987
5: -0.0024557, -0.0009568, -0.0023449, -0.0008343, -0.0016214, 0.0013881
6: -0.0060468, -0.0053530, -0.0060149, -0.0052870, -0.0007598, 0.0006619
7: -0.0032045, -0.0019478, -0.0033235, -0.0019272, -0.0012772, 0.0013757
8: -0.0044575, -0.0015528, -0.0042819, -0.0012914, -0.0031661, 0.0027290
9: 1.0004396, 1.0007663, 1.0004183, 1.0010285, -0.0005889, 0.0003480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0040554, -0.0017100, -0.0023529, 0.0023461
1: 0.0049768, 0.0066208, 0.0049773, 0.0066135, -0.0016367, 0.0016435
2: 0.0104779, 0.0149136, 0.0104874, 0.0149153, -0.0039589, 0.0039606
3: -0.0047365, -0.0027597, -0.0047332, -0.0027712, -0.0019653, 0.0019735
4: 0.0045490, 0.0052098, 0.0045508, 0.0052057, -0.0006455, 0.0006463
5: -0.0023801, -0.0008863, -0.0023792, -0.0008868, -0.0014933, 0.0014928
6: -0.0060301, -0.0053163, -0.0060288, -0.0053198, -0.0007104, 0.0007125
7: -0.0032520, -0.0019201, -0.0032559, -0.0019120, -0.0013400, 0.0013358
8: -0.0043444, -0.0014065, -0.0043454, -0.0014115, -0.0029330, 0.0029389
9: 1.0004286, 1.0009116, 1.0004292, 1.0008996, -0.0004710, 0.0004824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0040517, -0.0017120, -0.0023436, 0.0023755
1: 0.0049888, 0.0066256, 0.0049818, 0.0066135, -0.0016247, 0.0016438
2: 0.0104947, 0.0149935, 0.0104961, 0.0149118, -0.0039505, 0.0040311
3: -0.0047309, -0.0027582, -0.0047303, -0.0027712, -0.0019597, 0.0019721
4: 0.0045490, 0.0052096, 0.0045514, 0.0052056, -0.0006518, 0.0006445
5: -0.0024220, -0.0008967, -0.0023782, -0.0008918, -0.0015302, 0.0014815
6: -0.0060353, -0.0053176, -0.0060282, -0.0053204, -0.0007149, 0.0007105
7: -0.0032766, -0.0019509, -0.0032559, -0.0019205, -0.0013561, 0.0013051
8: -0.0044012, -0.0014181, -0.0043431, -0.0014176, -0.0029836, 0.0029250
9: 1.0004270, 1.0009035, 1.0004292, 1.0008957, -0.0004687, 0.0004743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0041599, -0.0017578, -0.0023052, 0.0024505
1: 0.0049768, 0.0066208, 0.0050008, 0.0066928, -0.0017160, 0.0016200
2: 0.0104779, 0.0149136, 0.0102953, 0.0148219, -0.0038221, 0.0041029
3: -0.0047365, -0.0027597, -0.0046989, -0.0026759, -0.0020606, 0.0019392
4: 0.0045490, 0.0052098, 0.0045628, 0.0052374, -0.0006690, 0.0006264
5: -0.0023801, -0.0008863, -0.0023460, -0.0008295, -0.0015506, 0.0014596
6: -0.0060301, -0.0053163, -0.0060155, -0.0052863, -0.0007439, 0.0006992
7: -0.0032520, -0.0019201, -0.0033235, -0.0019185, -0.0013335, 0.0014034
8: -0.0043444, -0.0014065, -0.0042841, -0.0012849, -0.0030596, 0.0028777
9: 1.0004286, 1.0009116, 1.0004183, 1.0010327, -0.0006042, 0.0004933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0041561, -0.0017597, -0.0022959, 0.0024799
1: 0.0049888, 0.0066256, 0.0050054, 0.0066928, -0.0017041, 0.0016202
2: 0.0104947, 0.0149935, 0.0103044, 0.0148185, -0.0038167, 0.0041728
3: -0.0047309, -0.0027582, -0.0046959, -0.0026759, -0.0020550, 0.0019378
4: 0.0045490, 0.0052096, 0.0045633, 0.0052374, -0.0006757, 0.0006247
5: -0.0024220, -0.0008967, -0.0023449, -0.0008343, -0.0015876, 0.0014482
6: -0.0060353, -0.0053176, -0.0060149, -0.0052870, -0.0007483, 0.0006972
7: -0.0032766, -0.0019509, -0.0033235, -0.0019272, -0.0013494, 0.0013726
8: -0.0044012, -0.0014181, -0.0042819, -0.0012914, -0.0031098, 0.0028638
9: 1.0004270, 1.0009035, 1.0004183, 1.0010285, -0.0006015, 0.0004852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0039673, -0.0016609, -0.0023789, 0.0022501
1: 0.0049900, 0.0066135, 0.0049500, 0.0065324, -0.0015423, 0.0016635
2: 0.0105249, 0.0149022, 0.0106434, 0.0150106, -0.0039830, 0.0037593
3: -0.0047243, -0.0027712, -0.0047712, -0.0028611, -0.0018632, 0.0020000
4: 0.0045530, 0.0052054, 0.0045374, 0.0051767, -0.0006157, 0.0006563
5: -0.0023756, -0.0009070, -0.0024164, -0.0009253, -0.0014502, 0.0015094
6: -0.0060265, -0.0053224, -0.0060438, -0.0053486, -0.0006779, 0.0007214
7: -0.0032559, -0.0019368, -0.0031820, -0.0018924, -0.0013635, 0.0012452
8: -0.0043370, -0.0014382, -0.0044103, -0.0015114, -0.0028256, 0.0029722
9: 1.0004292, 1.0008836, 1.0004410, 1.0007906, -0.0003614, 0.0004426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0039638, -0.0016624, -0.0023670, 0.0022745
1: 0.0050022, 0.0066194, 0.0049544, 0.0065324, -0.0015302, 0.0016650
2: 0.0105478, 0.0149702, 0.0106519, 0.0150077, -0.0039754, 0.0038266
3: -0.0047174, -0.0027692, -0.0047687, -0.0028611, -0.0018562, 0.0019995
4: 0.0045534, 0.0052053, 0.0045379, 0.0051767, -0.0006232, 0.0006548
5: -0.0024164, -0.0009202, -0.0024156, -0.0009299, -0.0014866, 0.0014954
6: -0.0060307, -0.0053243, -0.0060433, -0.0053493, -0.0006814, 0.0007190
7: -0.0032847, -0.0019655, -0.0031820, -0.0019015, -0.0013832, 0.0012165
8: -0.0043862, -0.0014543, -0.0044085, -0.0015175, -0.0028687, 0.0029542
9: 1.0004270, 1.0008731, 1.0004410, 1.0007870, -0.0003600, 0.0004321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0040793, -0.0017028, -0.0023369, 0.0023620
1: 0.0049900, 0.0066135, 0.0049647, 0.0066208, -0.0016308, 0.0016488
2: 0.0105249, 0.0149022, 0.0104387, 0.0149255, -0.0039325, 0.0039859
3: -0.0047243, -0.0027712, -0.0047450, -0.0027597, -0.0019646, 0.0019738
4: 0.0045530, 0.0052054, 0.0045469, 0.0052100, -0.0006467, 0.0006448
5: -0.0023756, -0.0009070, -0.0023835, -0.0008651, -0.0015104, 0.0014765
6: -0.0060265, -0.0053224, -0.0060322, -0.0053135, -0.0007131, 0.0007098
7: -0.0032559, -0.0019368, -0.0032520, -0.0018965, -0.0013595, 0.0013152
8: -0.0043370, -0.0014382, -0.0043518, -0.0013784, -0.0029586, 0.0029136
9: 1.0004292, 1.0008836, 1.0004286, 1.0009283, -0.0004991, 0.0004550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0040759, -0.0017044, -0.0023251, 0.0023865
1: 0.0050022, 0.0066194, 0.0049692, 0.0066208, -0.0016186, 0.0016502
2: 0.0105478, 0.0149702, 0.0104466, 0.0149226, -0.0039247, 0.0040526
3: -0.0047174, -0.0027692, -0.0047424, -0.0027597, -0.0019577, 0.0019732
4: 0.0045534, 0.0052053, 0.0045474, 0.0052100, -0.0006542, 0.0006434
5: -0.0024164, -0.0009202, -0.0023827, -0.0008698, -0.0015466, 0.0014625
6: -0.0060307, -0.0053243, -0.0060317, -0.0053141, -0.0007166, 0.0007074
7: -0.0032847, -0.0019655, -0.0032520, -0.0019049, -0.0013797, 0.0012865
8: -0.0043862, -0.0014543, -0.0043499, -0.0013841, -0.0030020, 0.0028957
9: 1.0004270, 1.0008731, 1.0004286, 1.0009246, -0.0004976, 0.0004445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0039673, -0.0016609, -0.0024840, 0.0022024
1: 0.0050150, 0.0066928, 0.0049500, 0.0065324, -0.0015174, 0.0017429
2: 0.0103310, 0.0148092, 0.0106434, 0.0150106, -0.0041904, 0.0036834
3: -0.0046896, -0.0026759, -0.0047712, -0.0028611, -0.0018284, 0.0020953
4: 0.0045649, 0.0052372, 0.0045374, 0.0051767, -0.0006002, 0.0006829
5: -0.0023423, -0.0008497, -0.0024164, -0.0009253, -0.0014169, 0.0015667
6: -0.0060133, -0.0052887, -0.0060438, -0.0053486, -0.0006646, 0.0007551
7: -0.0033235, -0.0019437, -0.0031820, -0.0018924, -0.0014311, 0.0012383
8: -0.0042760, -0.0013106, -0.0044103, -0.0015114, -0.0027646, 0.0030998
9: 1.0004183, 1.0010175, 1.0004410, 1.0007906, -0.0003723, 0.0005765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0039638, -0.0016624, -0.0024708, 0.0022257
1: 0.0050250, 0.0066997, 0.0049544, 0.0065324, -0.0015073, 0.0017453
2: 0.0103584, 0.0148763, 0.0106519, 0.0150077, -0.0041787, 0.0037407
3: -0.0046833, -0.0026736, -0.0047687, -0.0028611, -0.0018222, 0.0020951
4: 0.0045654, 0.0052370, 0.0045379, 0.0051767, -0.0006062, 0.0006814
5: -0.0023831, -0.0008633, -0.0024156, -0.0009299, -0.0014532, 0.0015523
6: -0.0060173, -0.0052911, -0.0060433, -0.0053493, -0.0006680, 0.0007522
7: -0.0033504, -0.0019719, -0.0031820, -0.0019015, -0.0014489, 0.0012100
8: -0.0043242, -0.0013300, -0.0044085, -0.0015175, -0.0028067, 0.0030784
9: 1.0004160, 1.0010043, 1.0004410, 1.0007870, -0.0003710, 0.0005634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0040793, -0.0017028, -0.0024421, 0.0023144
1: 0.0050150, 0.0066928, 0.0049647, 0.0066208, -0.0016058, 0.0017281
2: 0.0103310, 0.0148092, 0.0104387, 0.0149255, -0.0040747, 0.0038507
3: -0.0046896, -0.0026759, -0.0047450, -0.0027597, -0.0019299, 0.0020691
4: 0.0045649, 0.0052372, 0.0045469, 0.0052100, -0.0006271, 0.0006680
5: -0.0023423, -0.0008497, -0.0023835, -0.0008651, -0.0014771, 0.0015338
6: -0.0060133, -0.0052887, -0.0060322, -0.0053135, -0.0006998, 0.0007435
7: -0.0033235, -0.0019437, -0.0032520, -0.0018965, -0.0014270, 0.0013083
8: -0.0042760, -0.0013106, -0.0043518, -0.0013784, -0.0028976, 0.0030412
9: 1.0004183, 1.0010175, 1.0004286, 1.0009283, -0.0005100, 0.0005889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0040759, -0.0017044, -0.0024289, 0.0023377
1: 0.0050250, 0.0066997, 0.0049692, 0.0066208, -0.0015958, 0.0017305
2: 0.0103584, 0.0148763, 0.0104466, 0.0149226, -0.0040673, 0.0039114
3: -0.0046833, -0.0026736, -0.0047424, -0.0027597, -0.0019236, 0.0020687
4: 0.0045654, 0.0052370, 0.0045474, 0.0052100, -0.0006343, 0.0006666
5: -0.0023831, -0.0008633, -0.0023827, -0.0008698, -0.0015133, 0.0015194
6: -0.0060173, -0.0052911, -0.0060317, -0.0053141, -0.0007032, 0.0007406
7: -0.0033504, -0.0019719, -0.0032520, -0.0019049, -0.0014455, 0.0012801
8: -0.0043242, -0.0013300, -0.0043499, -0.0013841, -0.0029400, 0.0030199
9: 1.0004160, 1.0010043, 1.0004286, 1.0009246, -0.0005085, 0.0005758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0040554, -0.0017100, -0.0023297, 0.0023381
1: 0.0049900, 0.0066135, 0.0049773, 0.0066135, -0.0016235, 0.0016362
2: 0.0105249, 0.0149022, 0.0104874, 0.0149153, -0.0038131, 0.0038407
3: -0.0047243, -0.0027712, -0.0047332, -0.0027712, -0.0019531, 0.0019620
4: 0.0045530, 0.0052054, 0.0045508, 0.0052057, -0.0006474, 0.0006466
5: -0.0023756, -0.0009070, -0.0023792, -0.0008868, -0.0014888, 0.0014722
6: -0.0060265, -0.0053224, -0.0060288, -0.0053198, -0.0007068, 0.0007063
7: -0.0032559, -0.0019368, -0.0032559, -0.0019120, -0.0013439, 0.0013191
8: -0.0043370, -0.0014382, -0.0043454, -0.0014115, -0.0029255, 0.0029072
9: 1.0004292, 1.0008836, 1.0004292, 1.0008996, -0.0004704, 0.0004544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0040517, -0.0017120, -0.0023174, 0.0023623
1: 0.0050022, 0.0066194, 0.0049818, 0.0066135, -0.0016113, 0.0016376
2: 0.0105478, 0.0149702, 0.0104961, 0.0149118, -0.0038024, 0.0039103
3: -0.0047174, -0.0027692, -0.0047303, -0.0027712, -0.0019462, 0.0019611
4: 0.0045534, 0.0052053, 0.0045514, 0.0052056, -0.0006522, 0.0006452
5: -0.0024164, -0.0009202, -0.0023782, -0.0008918, -0.0015246, 0.0014580
6: -0.0060307, -0.0053243, -0.0060282, -0.0053204, -0.0007103, 0.0007038
7: -0.0032847, -0.0019655, -0.0032559, -0.0019205, -0.0013641, 0.0012904
8: -0.0043862, -0.0014543, -0.0043431, -0.0014176, -0.0029685, 0.0028889
9: 1.0004270, 1.0008731, 1.0004292, 1.0008957, -0.0004687, 0.0004439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0041599, -0.0017578, -0.0022819, 0.0024426
1: 0.0049900, 0.0066135, 0.0050008, 0.0066928, -0.0017028, 0.0016127
2: 0.0105249, 0.0149022, 0.0102953, 0.0148219, -0.0037590, 0.0040613
3: -0.0047243, -0.0027712, -0.0046989, -0.0026759, -0.0020484, 0.0019277
4: 0.0045530, 0.0052054, 0.0045628, 0.0052374, -0.0006785, 0.0006355
5: -0.0023756, -0.0009070, -0.0023460, -0.0008295, -0.0015461, 0.0014390
6: -0.0060265, -0.0053224, -0.0060155, -0.0052863, -0.0007403, 0.0006930
7: -0.0032559, -0.0019368, -0.0033235, -0.0019185, -0.0013375, 0.0013867
8: -0.0043370, -0.0014382, -0.0042841, -0.0012849, -0.0030521, 0.0028460
9: 1.0004292, 1.0008836, 1.0004183, 1.0010327, -0.0006036, 0.0004653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0041561, -0.0017597, -0.0022697, 0.0024667
1: 0.0050022, 0.0066194, 0.0050054, 0.0066928, -0.0016907, 0.0016140
2: 0.0105478, 0.0149702, 0.0103044, 0.0148185, -0.0037482, 0.0041307
3: -0.0047174, -0.0027692, -0.0046959, -0.0026759, -0.0020415, 0.0019267
4: 0.0045534, 0.0052053, 0.0045633, 0.0052374, -0.0006839, 0.0006341
5: -0.0024164, -0.0009202, -0.0023449, -0.0008343, -0.0015821, 0.0014247
6: -0.0060307, -0.0053243, -0.0060149, -0.0052870, -0.0007437, 0.0006905
7: -0.0032847, -0.0019655, -0.0033235, -0.0019272, -0.0013574, 0.0013580
8: -0.0043862, -0.0014543, -0.0042819, -0.0012914, -0.0030947, 0.0028276
9: 1.0004270, 1.0008731, 1.0004183, 1.0010285, -0.0006015, 0.0004548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0040554, -0.0017100, -0.0024348, 0.0022905
1: 0.0050150, 0.0066928, 0.0049773, 0.0066135, -0.0015985, 0.0017156
2: 0.0103310, 0.0148092, 0.0104874, 0.0149153, -0.0040343, 0.0037864
3: -0.0046896, -0.0026759, -0.0047332, -0.0027712, -0.0019184, 0.0020574
4: 0.0045649, 0.0052372, 0.0045508, 0.0052057, -0.0006361, 0.0006779
5: -0.0023423, -0.0008497, -0.0023792, -0.0008868, -0.0014555, 0.0015295
6: -0.0060133, -0.0052887, -0.0060288, -0.0053198, -0.0006935, 0.0007400
7: -0.0033235, -0.0019437, -0.0032559, -0.0019120, -0.0014115, 0.0013122
8: -0.0042760, -0.0013106, -0.0043454, -0.0014115, -0.0028646, 0.0030348
9: 1.0004183, 1.0010175, 1.0004292, 1.0008996, -0.0004812, 0.0005883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0040517, -0.0017120, -0.0024213, 0.0023136
1: 0.0050250, 0.0066997, 0.0049818, 0.0066135, -0.0015885, 0.0017179
2: 0.0103584, 0.0148763, 0.0104961, 0.0149118, -0.0040225, 0.0038482
3: -0.0046833, -0.0026736, -0.0047303, -0.0027712, -0.0019121, 0.0020567
4: 0.0045654, 0.0052370, 0.0045514, 0.0052056, -0.0006402, 0.0006764
5: -0.0023831, -0.0008633, -0.0023782, -0.0008918, -0.0014912, 0.0015149
6: -0.0060173, -0.0052911, -0.0060282, -0.0053204, -0.0006969, 0.0007371
7: -0.0033504, -0.0019719, -0.0032559, -0.0019205, -0.0014299, 0.0012840
8: -0.0043242, -0.0013300, -0.0043431, -0.0014176, -0.0029065, 0.0030131
9: 1.0004160, 1.0010043, 1.0004292, 1.0008957, -0.0004797, 0.0005752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0041599, -0.0017578, -0.0023871, 0.0023949
1: 0.0050150, 0.0066928, 0.0050008, 0.0066928, -0.0016779, 0.0016920
2: 0.0103310, 0.0148092, 0.0102953, 0.0148219, -0.0039027, 0.0039302
3: -0.0046896, -0.0026759, -0.0046989, -0.0026759, -0.0020137, 0.0020230
4: 0.0045649, 0.0052372, 0.0045628, 0.0052374, -0.0006543, 0.0006535
5: -0.0023423, -0.0008497, -0.0023460, -0.0008295, -0.0015128, 0.0014962
6: -0.0060133, -0.0052887, -0.0060155, -0.0052863, -0.0007270, 0.0007267
7: -0.0033235, -0.0019437, -0.0033235, -0.0019185, -0.0014050, 0.0013798
8: -0.0042760, -0.0013106, -0.0042841, -0.0012849, -0.0029911, 0.0029736
9: 1.0004183, 1.0010175, 1.0004183, 1.0010327, -0.0006144, 0.0005991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0041561, -0.0017597, -0.0023735, 0.0024179
1: 0.0050250, 0.0066997, 0.0050054, 0.0066928, -0.0016678, 0.0016943
2: 0.0103584, 0.0148763, 0.0103044, 0.0148185, -0.0038899, 0.0039955
3: -0.0046833, -0.0026736, -0.0046959, -0.0026759, -0.0020074, 0.0020223
4: 0.0045654, 0.0052370, 0.0045633, 0.0052374, -0.0006610, 0.0006521
5: -0.0023831, -0.0008633, -0.0023449, -0.0008343, -0.0015487, 0.0014817
6: -0.0060173, -0.0052911, -0.0060149, -0.0052870, -0.0007303, 0.0007238
7: -0.0033504, -0.0019719, -0.0033235, -0.0019272, -0.0014232, 0.0013516
8: -0.0043242, -0.0013300, -0.0042819, -0.0012914, -0.0030328, 0.0029519
9: 1.0004160, 1.0010043, 1.0004183, 1.0010285, -0.0006125, 0.0005860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.10 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004711, upper bound: 0.0004711
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004896, upper bound: 0.0004780
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004780, upper bound: 0.0004896
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005113, upper bound: 0.0005112
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004700, upper bound: 0.0004701
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004895, upper bound: 0.0004773
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004766, upper bound: 0.0004871
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005112, upper bound: 0.0005105
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004701, upper bound: 0.0004700
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004766
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004773, upper bound: 0.0004895
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005112
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004694, upper bound: 0.0004694
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004871, upper bound: 0.0004763
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0004763, upper bound: 0.0004871
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 9, lower bound: -0.0005105, upper bound: 0.0005105

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0039512, -0.0016674, -0.0022838, 0.0022838
1: 0.0049610, 0.0065324, 0.0049610, 0.0065324, -0.0015713, 0.0015713
2: 0.0106815, 0.0149987, 0.0106815, 0.0149987, -0.0037787, 0.0037787
3: -0.0047633, -0.0028611, -0.0047633, -0.0028611, -0.0019021, 0.0019021
4: 0.0045394, 0.0051765, 0.0045394, 0.0051765, -0.0006155, 0.0006155
5: -0.0024131, -0.0009460, -0.0024131, -0.0009460, -0.0014671, 0.0014671
6: -0.0060418, -0.0053514, -0.0060418, -0.0053514, -0.0006904, 0.0006904
7: -0.0031820, -0.0019151, -0.0031820, -0.0019151, -0.0012669, 0.0012669
8: -0.0044025, -0.0015385, -0.0044025, -0.0015385, -0.0028640, 0.0028640
9: 1.0004410, 1.0007743, 1.0004410, 1.0007743, -0.0003333, 0.0003333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004688, upper bound: 0.0004702
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0039428, -0.0016366, -0.0023145, 0.0022754
1: 0.0049610, 0.0065324, 0.0049754, 0.0065367, -0.0015756, 0.0015570
2: 0.0106815, 0.0149987, 0.0107018, 0.0150746, -0.0038585, 0.0037647
3: -0.0047633, -0.0028611, -0.0047573, -0.0028598, -0.0019035, 0.0018961
4: 0.0045394, 0.0051765, 0.0045393, 0.0051763, -0.0006153, 0.0006163
5: -0.0024131, -0.0009460, -0.0024557, -0.0009568, -0.0014563, 0.0015097
6: -0.0060418, -0.0053514, -0.0060468, -0.0053530, -0.0006888, 0.0006954
7: -0.0031820, -0.0019151, -0.0032045, -0.0019478, -0.0012341, 0.0012894
8: -0.0044025, -0.0015385, -0.0044575, -0.0015528, -0.0028497, 0.0029190
9: 1.0004410, 1.0007743, 1.0004396, 1.0007663, -0.0003253, 0.0003346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004688, upper bound: 0.0004702
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0039512, -0.0016674, -0.0022754, 0.0023145
1: 0.0049754, 0.0065367, 0.0049610, 0.0065324, -0.0015570, 0.0015756
2: 0.0107018, 0.0150746, 0.0106815, 0.0149987, -0.0037647, 0.0038585
3: -0.0047573, -0.0028598, -0.0047633, -0.0028611, -0.0018961, 0.0019035
4: 0.0045393, 0.0051763, 0.0045394, 0.0051765, -0.0006163, 0.0006153
5: -0.0024557, -0.0009568, -0.0024131, -0.0009460, -0.0015097, 0.0014563
6: -0.0060468, -0.0053530, -0.0060418, -0.0053514, -0.0006954, 0.0006888
7: -0.0032045, -0.0019478, -0.0031820, -0.0019151, -0.0012894, 0.0012341
8: -0.0044575, -0.0015528, -0.0044025, -0.0015385, -0.0029190, 0.0028497
9: 1.0004396, 1.0007663, 1.0004410, 1.0007743, -0.0003346, 0.0003253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004673, upper bound: 0.0004676
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0039428, -0.0016366, -0.0023061, 0.0023061
1: 0.0049754, 0.0065367, 0.0049754, 0.0065367, -0.0015613, 0.0015613
2: 0.0107018, 0.0150746, 0.0107018, 0.0150746, -0.0038204, 0.0038204
3: -0.0047573, -0.0028598, -0.0047573, -0.0028598, -0.0018975, 0.0018975
4: 0.0045393, 0.0051763, 0.0045393, 0.0051763, -0.0006210, 0.0006210
5: -0.0024557, -0.0009568, -0.0024557, -0.0009568, -0.0014989, 0.0014989
6: -0.0060468, -0.0053530, -0.0060468, -0.0053530, -0.0006938, 0.0006938
7: -0.0032045, -0.0019478, -0.0032045, -0.0019478, -0.0012567, 0.0012567
8: -0.0044575, -0.0015528, -0.0044575, -0.0015528, -0.0029046, 0.0029046
9: 1.0004396, 1.0007663, 1.0004396, 1.0007663, -0.0003266, 0.0003266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004673, upper bound: 0.0004676
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0040630, -0.0017093, -0.0022418, 0.0023956
1: 0.0049610, 0.0065324, 0.0049768, 0.0066208, -0.0016597, 0.0015556
2: 0.0106815, 0.0149987, 0.0104779, 0.0149136, -0.0037274, 0.0040067
3: -0.0047633, -0.0028611, -0.0047365, -0.0027597, -0.0020036, 0.0018754
4: 0.0045394, 0.0051765, 0.0045490, 0.0052098, -0.0006465, 0.0006036
5: -0.0024131, -0.0009460, -0.0023801, -0.0008863, -0.0015267, 0.0014341
6: -0.0060418, -0.0053514, -0.0060301, -0.0053163, -0.0007255, 0.0006788
7: -0.0031820, -0.0019151, -0.0032520, -0.0019201, -0.0012619, 0.0013369
8: -0.0044025, -0.0015385, -0.0043444, -0.0014065, -0.0029960, 0.0028060
9: 1.0004410, 1.0007743, 1.0004286, 1.0009116, -0.0004706, 0.0003457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004784
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004760
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0040556, -0.0016762, -0.0022750, 0.0023882
1: 0.0049610, 0.0065324, 0.0049888, 0.0066256, -0.0016646, 0.0015436
2: 0.0106815, 0.0149987, 0.0104947, 0.0149935, -0.0038062, 0.0039882
3: -0.0047633, -0.0028611, -0.0047309, -0.0027582, -0.0020051, 0.0018698
4: 0.0045394, 0.0051765, 0.0045490, 0.0052096, -0.0006464, 0.0006043
5: -0.0024131, -0.0009460, -0.0024220, -0.0008967, -0.0015164, 0.0014760
6: -0.0060418, -0.0053514, -0.0060353, -0.0053176, -0.0007241, 0.0006840
7: -0.0031820, -0.0019151, -0.0032766, -0.0019509, -0.0012311, 0.0013616
8: -0.0044025, -0.0015385, -0.0044012, -0.0014181, -0.0029844, 0.0028627
9: 1.0004410, 1.0007743, 1.0004270, 1.0009035, -0.0004625, 0.0003473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004784
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004760
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0040630, -0.0017093, -0.0022334, 0.0024263
1: 0.0049754, 0.0065367, 0.0049768, 0.0066208, -0.0016454, 0.0015598
2: 0.0107018, 0.0150746, 0.0104779, 0.0149136, -0.0037135, 0.0040864
3: -0.0047573, -0.0028598, -0.0047365, -0.0027597, -0.0019976, 0.0018767
4: 0.0045393, 0.0051763, 0.0045490, 0.0052098, -0.0006474, 0.0006035
5: -0.0024557, -0.0009568, -0.0023801, -0.0008863, -0.0015694, 0.0014232
6: -0.0060468, -0.0053530, -0.0060301, -0.0053163, -0.0007305, 0.0006772
7: -0.0032045, -0.0019478, -0.0032520, -0.0019201, -0.0012844, 0.0013042
8: -0.0044575, -0.0015528, -0.0043444, -0.0014065, -0.0030510, 0.0027916
9: 1.0004396, 1.0007663, 1.0004286, 1.0009116, -0.0004719, 0.0003377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004757
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004848, upper bound: 0.0004737
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0040556, -0.0016762, -0.0022666, 0.0024189
1: 0.0049754, 0.0065367, 0.0049888, 0.0066256, -0.0016502, 0.0015479
2: 0.0107018, 0.0150746, 0.0104947, 0.0149935, -0.0037691, 0.0040455
3: -0.0047573, -0.0028598, -0.0047309, -0.0027582, -0.0019991, 0.0018711
4: 0.0045393, 0.0051763, 0.0045490, 0.0052096, -0.0006518, 0.0006086
5: -0.0024557, -0.0009568, -0.0024220, -0.0008967, -0.0015590, 0.0014652
6: -0.0060468, -0.0053530, -0.0060353, -0.0053176, -0.0007291, 0.0006824
7: -0.0032045, -0.0019478, -0.0032766, -0.0019509, -0.0012536, 0.0013288
8: -0.0044575, -0.0015528, -0.0044012, -0.0014181, -0.0030394, 0.0028484
9: 1.0004396, 1.0007663, 1.0004270, 1.0009035, -0.0004638, 0.0003393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004757
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004848, upper bound: 0.0004737
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0039512, -0.0016674, -0.0023956, 0.0022418
1: 0.0049768, 0.0066208, 0.0049610, 0.0065324, -0.0015556, 0.0016597
2: 0.0104779, 0.0149136, 0.0106815, 0.0149987, -0.0040067, 0.0037274
3: -0.0047365, -0.0027597, -0.0047633, -0.0028611, -0.0018754, 0.0020036
4: 0.0045490, 0.0052098, 0.0045394, 0.0051765, -0.0006036, 0.0006465
5: -0.0023801, -0.0008863, -0.0024131, -0.0009460, -0.0014341, 0.0015267
6: -0.0060301, -0.0053163, -0.0060418, -0.0053514, -0.0006788, 0.0007255
7: -0.0032520, -0.0019201, -0.0031820, -0.0019151, -0.0013369, 0.0012619
8: -0.0043444, -0.0014065, -0.0044025, -0.0015385, -0.0028060, 0.0029960
9: 1.0004286, 1.0009116, 1.0004410, 1.0007743, -0.0003457, 0.0004706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003636, upper bound: 0.0003778
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003293, upper bound: 0.0003380
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0039428, -0.0016366, -0.0024263, 0.0022334
1: 0.0049768, 0.0066208, 0.0049754, 0.0065367, -0.0015598, 0.0016454
2: 0.0104779, 0.0149136, 0.0107018, 0.0150746, -0.0040864, 0.0037135
3: -0.0047365, -0.0027597, -0.0047573, -0.0028598, -0.0018767, 0.0019976
4: 0.0045490, 0.0052098, 0.0045393, 0.0051763, -0.0006035, 0.0006474
5: -0.0023801, -0.0008863, -0.0024557, -0.0009568, -0.0014232, 0.0015694
6: -0.0060301, -0.0053163, -0.0060468, -0.0053530, -0.0006772, 0.0007305
7: -0.0032520, -0.0019201, -0.0032045, -0.0019478, -0.0013042, 0.0012844
8: -0.0043444, -0.0014065, -0.0044575, -0.0015528, -0.0027916, 0.0030510
9: 1.0004286, 1.0009116, 1.0004396, 1.0007663, -0.0003377, 0.0004719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003636, upper bound: 0.0003778
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003293, upper bound: 0.0003380
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0039512, -0.0016674, -0.0023882, 0.0022750
1: 0.0049888, 0.0066256, 0.0049610, 0.0065324, -0.0015436, 0.0016646
2: 0.0104947, 0.0149935, 0.0106815, 0.0149987, -0.0039882, 0.0038062
3: -0.0047309, -0.0027582, -0.0047633, -0.0028611, -0.0018698, 0.0020051
4: 0.0045490, 0.0052096, 0.0045394, 0.0051765, -0.0006043, 0.0006464
5: -0.0024220, -0.0008967, -0.0024131, -0.0009460, -0.0014760, 0.0015164
6: -0.0060353, -0.0053176, -0.0060418, -0.0053514, -0.0006840, 0.0007241
7: -0.0032766, -0.0019509, -0.0031820, -0.0019151, -0.0013616, 0.0012311
8: -0.0044012, -0.0014181, -0.0044025, -0.0015385, -0.0028627, 0.0029844
9: 1.0004270, 1.0009035, 1.0004410, 1.0007743, -0.0003473, 0.0004625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003621, upper bound: 0.0003766
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003226, upper bound: 0.0003253
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0039428, -0.0016366, -0.0024189, 0.0022666
1: 0.0049888, 0.0066256, 0.0049754, 0.0065367, -0.0015479, 0.0016502
2: 0.0104947, 0.0149935, 0.0107018, 0.0150746, -0.0040455, 0.0037691
3: -0.0047309, -0.0027582, -0.0047573, -0.0028598, -0.0018711, 0.0019991
4: 0.0045490, 0.0052096, 0.0045393, 0.0051763, -0.0006086, 0.0006518
5: -0.0024220, -0.0008967, -0.0024557, -0.0009568, -0.0014652, 0.0015590
6: -0.0060353, -0.0053176, -0.0060468, -0.0053530, -0.0006824, 0.0007291
7: -0.0032766, -0.0019509, -0.0032045, -0.0019478, -0.0013288, 0.0012536
8: -0.0044012, -0.0014181, -0.0044575, -0.0015528, -0.0028484, 0.0030394
9: 1.0004270, 1.0009035, 1.0004396, 1.0007663, -0.0003393, 0.0004638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003621, upper bound: 0.0003766
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003226, upper bound: 0.0003253
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0040630, -0.0017093, -0.0023536, 0.0023536
1: 0.0049768, 0.0066208, 0.0049768, 0.0066208, -0.0016440, 0.0016440
2: 0.0104779, 0.0149136, 0.0104779, 0.0149136, -0.0038805, 0.0038805
3: -0.0047365, -0.0027597, -0.0047365, -0.0027597, -0.0019768, 0.0019768
4: 0.0045490, 0.0052098, 0.0045490, 0.0052098, -0.0006270, 0.0006270
5: -0.0023801, -0.0008863, -0.0023801, -0.0008863, -0.0014937, 0.0014937
6: -0.0060301, -0.0053163, -0.0060301, -0.0053163, -0.0007139, 0.0007139
7: -0.0032520, -0.0019201, -0.0032520, -0.0019201, -0.0013319, 0.0013319
8: -0.0043444, -0.0014065, -0.0043444, -0.0014065, -0.0029380, 0.0029380
9: 1.0004286, 1.0009116, 1.0004286, 1.0009116, -0.0004830, 0.0004830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003520, upper bound: 0.0003586
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0040556, -0.0016762, -0.0023868, 0.0023462
1: 0.0049768, 0.0066208, 0.0049888, 0.0066256, -0.0016488, 0.0016320
2: 0.0104779, 0.0149136, 0.0104947, 0.0149935, -0.0039588, 0.0038646
3: -0.0047365, -0.0027597, -0.0047309, -0.0027582, -0.0019783, 0.0019712
4: 0.0045490, 0.0052098, 0.0045490, 0.0052096, -0.0006268, 0.0006278
5: -0.0023801, -0.0008863, -0.0024220, -0.0008967, -0.0014834, 0.0015356
6: -0.0060301, -0.0053163, -0.0060353, -0.0053176, -0.0007125, 0.0007190
7: -0.0032520, -0.0019201, -0.0032766, -0.0019509, -0.0013011, 0.0013565
8: -0.0043444, -0.0014065, -0.0044012, -0.0014181, -0.0029263, 0.0029947
9: 1.0004286, 1.0009116, 1.0004270, 1.0009035, -0.0004749, 0.0004846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003520, upper bound: 0.0003586
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0040630, -0.0017093, -0.0023462, 0.0023868
1: 0.0049888, 0.0066256, 0.0049768, 0.0066208, -0.0016320, 0.0016488
2: 0.0104947, 0.0149935, 0.0104779, 0.0149136, -0.0038646, 0.0039588
3: -0.0047309, -0.0027582, -0.0047365, -0.0027597, -0.0019712, 0.0019783
4: 0.0045490, 0.0052096, 0.0045490, 0.0052098, -0.0006278, 0.0006268
5: -0.0024220, -0.0008967, -0.0023801, -0.0008863, -0.0015356, 0.0014834
6: -0.0060353, -0.0053176, -0.0060301, -0.0053163, -0.0007190, 0.0007125
7: -0.0032766, -0.0019509, -0.0032520, -0.0019201, -0.0013565, 0.0013011
8: -0.0044012, -0.0014181, -0.0043444, -0.0014065, -0.0029947, 0.0029263
9: 1.0004270, 1.0009035, 1.0004286, 1.0009116, -0.0004846, 0.0004749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003449, upper bound: 0.0003448
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0040556, -0.0016762, -0.0023794, 0.0023794
1: 0.0049888, 0.0066256, 0.0049888, 0.0066256, -0.0016368, 0.0016368
2: 0.0104947, 0.0149935, 0.0104947, 0.0149935, -0.0039215, 0.0039215
3: -0.0047309, -0.0027582, -0.0047309, -0.0027582, -0.0019727, 0.0019727
4: 0.0045490, 0.0052096, 0.0045490, 0.0052096, -0.0006323, 0.0006323
5: -0.0024220, -0.0008967, -0.0024220, -0.0008967, -0.0015253, 0.0015253
6: -0.0060353, -0.0053176, -0.0060353, -0.0053176, -0.0007177, 0.0007177
7: -0.0032766, -0.0019509, -0.0032766, -0.0019509, -0.0013258, 0.0013258
8: -0.0044012, -0.0014181, -0.0044012, -0.0014181, -0.0029831, 0.0029831
9: 1.0004270, 1.0009035, 1.0004270, 1.0009035, -0.0004765, 0.0004765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003449, upper bound: 0.0003448
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0040397, -0.0017173, -0.0022339, 0.0023723
1: 0.0049610, 0.0065324, 0.0049900, 0.0066135, -0.0016525, 0.0015423
2: 0.0106815, 0.0149987, 0.0105249, 0.0149022, -0.0037202, 0.0039739
3: -0.0047633, -0.0028611, -0.0047243, -0.0027712, -0.0019921, 0.0018632
4: 0.0045394, 0.0051765, 0.0045530, 0.0052054, -0.0006530, 0.0006117
5: -0.0024131, -0.0009460, -0.0023756, -0.0009070, -0.0015061, 0.0014296
6: -0.0060418, -0.0053514, -0.0060265, -0.0053224, -0.0007193, 0.0006752
7: -0.0031820, -0.0019151, -0.0032559, -0.0019368, -0.0012452, 0.0013408
8: -0.0044025, -0.0015385, -0.0043370, -0.0014382, -0.0029644, 0.0027985
9: 1.0004410, 1.0007743, 1.0004292, 1.0008836, -0.0004426, 0.0003451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004682, upper bound: 0.0004696
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004658, upper bound: 0.0004672
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0040294, -0.0016894, -0.0022618, 0.0023620
1: 0.0049610, 0.0065324, 0.0050022, 0.0066194, -0.0016583, 0.0015302
2: 0.0106815, 0.0149987, 0.0105478, 0.0149702, -0.0037952, 0.0039546
3: -0.0047633, -0.0028611, -0.0047174, -0.0027692, -0.0019941, 0.0018562
4: 0.0045394, 0.0051765, 0.0045534, 0.0052053, -0.0006529, 0.0006122
5: -0.0024131, -0.0009460, -0.0024164, -0.0009202, -0.0014929, 0.0014704
6: -0.0060418, -0.0053514, -0.0060307, -0.0053243, -0.0007174, 0.0006793
7: -0.0031820, -0.0019151, -0.0032847, -0.0019655, -0.0012165, 0.0013696
8: -0.0044025, -0.0015385, -0.0043862, -0.0014543, -0.0029482, 0.0028477
9: 1.0004410, 1.0007743, 1.0004270, 1.0008731, -0.0004321, 0.0003473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004682, upper bound: 0.0004696
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004658, upper bound: 0.0004672
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0040397, -0.0017173, -0.0022255, 0.0024031
1: 0.0049754, 0.0065367, 0.0049900, 0.0066135, -0.0016381, 0.0015466
2: 0.0107018, 0.0150746, 0.0105249, 0.0149022, -0.0037062, 0.0040537
3: -0.0047573, -0.0028598, -0.0047243, -0.0027712, -0.0019861, 0.0018645
4: 0.0045393, 0.0051763, 0.0045530, 0.0052054, -0.0006539, 0.0006115
5: -0.0024557, -0.0009568, -0.0023756, -0.0009070, -0.0015487, 0.0014187
6: -0.0060468, -0.0053530, -0.0060265, -0.0053224, -0.0007243, 0.0006736
7: -0.0032045, -0.0019478, -0.0032559, -0.0019368, -0.0012677, 0.0013081
8: -0.0044575, -0.0015528, -0.0043370, -0.0014382, -0.0030193, 0.0027841
9: 1.0004396, 1.0007663, 1.0004292, 1.0008836, -0.0004439, 0.0003371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004666, upper bound: 0.0004670
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004650, upper bound: 0.0004652
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0040294, -0.0016894, -0.0022534, 0.0023928
1: 0.0049754, 0.0065367, 0.0050022, 0.0066194, -0.0016440, 0.0015345
2: 0.0107018, 0.0150746, 0.0105478, 0.0149702, -0.0037586, 0.0040128
3: -0.0047573, -0.0028598, -0.0047174, -0.0027692, -0.0019881, 0.0018576
4: 0.0045393, 0.0051763, 0.0045534, 0.0052053, -0.0006592, 0.0006177
5: -0.0024557, -0.0009568, -0.0024164, -0.0009202, -0.0015355, 0.0014596
6: -0.0060468, -0.0053530, -0.0060307, -0.0053243, -0.0007224, 0.0006777
7: -0.0032045, -0.0019478, -0.0032847, -0.0019655, -0.0012390, 0.0013368
8: -0.0044575, -0.0015528, -0.0043862, -0.0014543, -0.0030032, 0.0028333
9: 1.0004396, 1.0007663, 1.0004270, 1.0008731, -0.0004334, 0.0003393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004666, upper bound: 0.0004670
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004650, upper bound: 0.0004652
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0041449, -0.0017649, -0.0021862, 0.0024775
1: 0.0049610, 0.0065324, 0.0050150, 0.0066928, -0.0017318, 0.0015174
2: 0.0106815, 0.0149987, 0.0103310, 0.0148092, -0.0036443, 0.0041813
3: -0.0047633, -0.0028611, -0.0046896, -0.0026759, -0.0020874, 0.0018284
4: 0.0045394, 0.0051765, 0.0045649, 0.0052372, -0.0006796, 0.0005962
5: -0.0024131, -0.0009460, -0.0023423, -0.0008497, -0.0015634, 0.0013963
6: -0.0060418, -0.0053514, -0.0060133, -0.0052887, -0.0007530, 0.0006619
7: -0.0031820, -0.0019151, -0.0033235, -0.0019437, -0.0012383, 0.0014084
8: -0.0044025, -0.0015385, -0.0042760, -0.0013106, -0.0030920, 0.0027375
9: 1.0004410, 1.0007743, 1.0004183, 1.0010175, -0.0005765, 0.0003560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004780
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004756
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016674, -0.0041332, -0.0017381, -0.0022130, 0.0024658
1: 0.0049610, 0.0065324, 0.0050250, 0.0066997, -0.0017387, 0.0015073
2: 0.0106815, 0.0149987, 0.0103584, 0.0148763, -0.0037093, 0.0041556
3: -0.0047633, -0.0028611, -0.0046833, -0.0026736, -0.0020897, 0.0018222
4: 0.0045394, 0.0051765, 0.0045654, 0.0052370, -0.0006795, 0.0005965
5: -0.0024131, -0.0009460, -0.0023831, -0.0008633, -0.0015498, 0.0014371
6: -0.0060418, -0.0053514, -0.0060173, -0.0052911, -0.0007506, 0.0006659
7: -0.0031820, -0.0019151, -0.0033504, -0.0019719, -0.0012100, 0.0014353
8: -0.0044025, -0.0015385, -0.0043242, -0.0013300, -0.0030725, 0.0027857
9: 1.0004410, 1.0007743, 1.0004160, 1.0010043, -0.0005634, 0.0003582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004780
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004756
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0041449, -0.0017649, -0.0021779, 0.0025082
1: 0.0049754, 0.0065367, 0.0050150, 0.0066928, -0.0017175, 0.0015217
2: 0.0107018, 0.0150746, 0.0103310, 0.0148092, -0.0036303, 0.0042611
3: -0.0047573, -0.0028598, -0.0046896, -0.0026759, -0.0020814, 0.0018298
4: 0.0045393, 0.0051763, 0.0045649, 0.0052372, -0.0006804, 0.0005960
5: -0.0024557, -0.0009568, -0.0023423, -0.0008497, -0.0016060, 0.0013854
6: -0.0060468, -0.0053530, -0.0060133, -0.0052887, -0.0007580, 0.0006603
7: -0.0032045, -0.0019478, -0.0033235, -0.0019437, -0.0012608, 0.0013757
8: -0.0044575, -0.0015528, -0.0042760, -0.0013106, -0.0031469, 0.0027232
9: 1.0004396, 1.0007663, 1.0004183, 1.0010175, -0.0005778, 0.0003480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004754
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004734
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016366, -0.0041332, -0.0017381, -0.0022047, 0.0024966
1: 0.0049754, 0.0065367, 0.0050250, 0.0066997, -0.0017243, 0.0015116
2: 0.0107018, 0.0150746, 0.0103584, 0.0148763, -0.0036761, 0.0042162
3: -0.0047573, -0.0028598, -0.0046833, -0.0026736, -0.0020837, 0.0018235
4: 0.0045393, 0.0051763, 0.0045654, 0.0052370, -0.0006855, 0.0006008
5: -0.0024557, -0.0009568, -0.0023831, -0.0008633, -0.0015924, 0.0014262
6: -0.0060468, -0.0053530, -0.0060173, -0.0052911, -0.0007556, 0.0006643
7: -0.0032045, -0.0019478, -0.0033504, -0.0019719, -0.0012325, 0.0014026
8: -0.0044575, -0.0015528, -0.0043242, -0.0013300, -0.0031275, 0.0027713
9: 1.0004396, 1.0007663, 1.0004160, 1.0010043, -0.0005647, 0.0003502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004754
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004734
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0040397, -0.0017173, -0.0023457, 0.0023304
1: 0.0049768, 0.0066208, 0.0049900, 0.0066135, -0.0016367, 0.0016308
2: 0.0104779, 0.0149136, 0.0105249, 0.0149022, -0.0039481, 0.0039226
3: -0.0047365, -0.0027597, -0.0047243, -0.0027712, -0.0019653, 0.0019646
4: 0.0045490, 0.0052098, 0.0045530, 0.0052054, -0.0006412, 0.0006427
5: -0.0023801, -0.0008863, -0.0023756, -0.0009070, -0.0014731, 0.0014892
6: -0.0060301, -0.0053163, -0.0060265, -0.0053224, -0.0007077, 0.0007103
7: -0.0032520, -0.0019201, -0.0032559, -0.0019368, -0.0013152, 0.0013358
8: -0.0043444, -0.0014065, -0.0043370, -0.0014382, -0.0029063, 0.0029305
9: 1.0004286, 1.0009116, 1.0004292, 1.0008836, -0.0004550, 0.0004824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0040294, -0.0016894, -0.0023736, 0.0023201
1: 0.0049768, 0.0066208, 0.0050022, 0.0066194, -0.0016426, 0.0016186
2: 0.0104779, 0.0149136, 0.0105478, 0.0149702, -0.0040231, 0.0039033
3: -0.0047365, -0.0027597, -0.0047174, -0.0027692, -0.0019673, 0.0019577
4: 0.0045490, 0.0052098, 0.0045534, 0.0052053, -0.0006411, 0.0006433
5: -0.0023801, -0.0008863, -0.0024164, -0.0009202, -0.0014599, 0.0015301
6: -0.0060301, -0.0053163, -0.0060307, -0.0053243, -0.0007058, 0.0007144
7: -0.0032520, -0.0019201, -0.0032847, -0.0019655, -0.0012865, 0.0013645
8: -0.0043444, -0.0014065, -0.0043862, -0.0014543, -0.0028902, 0.0029797
9: 1.0004286, 1.0009116, 1.0004270, 1.0008731, -0.0004445, 0.0004846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0040397, -0.0017173, -0.0023383, 0.0023635
1: 0.0049888, 0.0066256, 0.0049900, 0.0066135, -0.0016247, 0.0016356
2: 0.0104947, 0.0149935, 0.0105249, 0.0149022, -0.0039297, 0.0040014
3: -0.0047309, -0.0027582, -0.0047243, -0.0027712, -0.0019597, 0.0019661
4: 0.0045490, 0.0052096, 0.0045530, 0.0052054, -0.0006418, 0.0006426
5: -0.0024220, -0.0008967, -0.0023756, -0.0009070, -0.0015150, 0.0014789
6: -0.0060353, -0.0053176, -0.0060265, -0.0053224, -0.0007129, 0.0007089
7: -0.0032766, -0.0019509, -0.0032559, -0.0019368, -0.0013398, 0.0013051
8: -0.0044012, -0.0014181, -0.0043370, -0.0014382, -0.0029631, 0.0029189
9: 1.0004270, 1.0009035, 1.0004292, 1.0008836, -0.0004566, 0.0004743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003170, upper bound: 0.0003209
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0040294, -0.0016894, -0.0023662, 0.0023532
1: 0.0049888, 0.0066256, 0.0050022, 0.0066194, -0.0016306, 0.0016235
2: 0.0104947, 0.0149935, 0.0105478, 0.0149702, -0.0039837, 0.0039615
3: -0.0047309, -0.0027582, -0.0047174, -0.0027692, -0.0019617, 0.0019592
4: 0.0045490, 0.0052096, 0.0045534, 0.0052053, -0.0006468, 0.0006486
5: -0.0024220, -0.0008967, -0.0024164, -0.0009202, -0.0015018, 0.0015197
6: -0.0060353, -0.0053176, -0.0060307, -0.0053243, -0.0007110, 0.0007130
7: -0.0032766, -0.0019509, -0.0032847, -0.0019655, -0.0013112, 0.0013338
8: -0.0044012, -0.0014181, -0.0043862, -0.0014543, -0.0029469, 0.0029681
9: 1.0004270, 1.0009035, 1.0004270, 1.0008731, -0.0004461, 0.0004765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003170, upper bound: 0.0003209
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0041449, -0.0017649, -0.0022980, 0.0024355
1: 0.0049768, 0.0066208, 0.0050150, 0.0066928, -0.0017160, 0.0016058
2: 0.0104779, 0.0149136, 0.0103310, 0.0148092, -0.0038113, 0.0040652
3: -0.0047365, -0.0027597, -0.0046896, -0.0026759, -0.0020606, 0.0019299
4: 0.0045490, 0.0052098, 0.0045649, 0.0052372, -0.0006647, 0.0006230
5: -0.0023801, -0.0008863, -0.0023423, -0.0008497, -0.0015304, 0.0014559
6: -0.0060301, -0.0053163, -0.0060133, -0.0052887, -0.0007414, 0.0006970
7: -0.0032520, -0.0019201, -0.0033235, -0.0019437, -0.0013083, 0.0014034
8: -0.0043444, -0.0014065, -0.0042760, -0.0013106, -0.0030339, 0.0028695
9: 1.0004286, 1.0009116, 1.0004183, 1.0010175, -0.0005889, 0.0004933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003580
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040630, -0.0017093, -0.0041332, -0.0017381, -0.0023248, 0.0024239
1: 0.0049768, 0.0066208, 0.0050250, 0.0066997, -0.0017229, 0.0015958
2: 0.0104779, 0.0149136, 0.0103584, 0.0148763, -0.0038801, 0.0040445
3: -0.0047365, -0.0027597, -0.0046833, -0.0026736, -0.0020629, 0.0019236
4: 0.0045490, 0.0052098, 0.0045654, 0.0052370, -0.0006646, 0.0006236
5: -0.0023801, -0.0008863, -0.0023831, -0.0008633, -0.0015168, 0.0014967
6: -0.0060301, -0.0053163, -0.0060173, -0.0052911, -0.0007390, 0.0007010
7: -0.0032520, -0.0019201, -0.0033504, -0.0019719, -0.0012801, 0.0014303
8: -0.0043444, -0.0014065, -0.0043242, -0.0013300, -0.0030144, 0.0029177
9: 1.0004286, 1.0009116, 1.0004160, 1.0010043, -0.0005758, 0.0004956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003580
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0041449, -0.0017649, -0.0022906, 0.0024687
1: 0.0049888, 0.0066256, 0.0050150, 0.0066928, -0.0017041, 0.0016106
2: 0.0104947, 0.0149935, 0.0103310, 0.0148092, -0.0037954, 0.0041435
3: -0.0047309, -0.0027582, -0.0046896, -0.0026759, -0.0020550, 0.0019314
4: 0.0045490, 0.0052096, 0.0045649, 0.0052372, -0.0006655, 0.0006229
5: -0.0024220, -0.0008967, -0.0023423, -0.0008497, -0.0015723, 0.0014456
6: -0.0060353, -0.0053176, -0.0060133, -0.0052887, -0.0007466, 0.0006956
7: -0.0032766, -0.0019509, -0.0033235, -0.0019437, -0.0013330, 0.0013726
8: -0.0044012, -0.0014181, -0.0042760, -0.0013106, -0.0030907, 0.0028579
9: 1.0004270, 1.0009035, 1.0004183, 1.0010175, -0.0005904, 0.0004852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003447, upper bound: 0.0003442
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040556, -0.0016762, -0.0041332, -0.0017381, -0.0023174, 0.0024571
1: 0.0049888, 0.0066256, 0.0050250, 0.0066997, -0.0017109, 0.0016006
2: 0.0104947, 0.0149935, 0.0103584, 0.0148763, -0.0038460, 0.0041035
3: -0.0047309, -0.0027582, -0.0046833, -0.0026736, -0.0020573, 0.0019251
4: 0.0045490, 0.0052096, 0.0045654, 0.0052370, -0.0006707, 0.0006290
5: -0.0024220, -0.0008967, -0.0023831, -0.0008633, -0.0015587, 0.0014864
6: -0.0060353, -0.0053176, -0.0060173, -0.0052911, -0.0007442, 0.0006996
7: -0.0032766, -0.0019509, -0.0033504, -0.0019719, -0.0013047, 0.0013995
8: -0.0044012, -0.0014181, -0.0043242, -0.0013300, -0.0030712, 0.0029061
9: 1.0004270, 1.0009035, 1.0004160, 1.0010043, -0.0005773, 0.0004874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003447, upper bound: 0.0003442
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0039512, -0.0016674, -0.0023723, 0.0022339
1: 0.0049900, 0.0066135, 0.0049610, 0.0065324, -0.0015423, 0.0016525
2: 0.0105249, 0.0149022, 0.0106815, 0.0149987, -0.0039739, 0.0037202
3: -0.0047243, -0.0027712, -0.0047633, -0.0028611, -0.0018632, 0.0019921
4: 0.0045530, 0.0052054, 0.0045394, 0.0051765, -0.0006117, 0.0006530
5: -0.0023756, -0.0009070, -0.0024131, -0.0009460, -0.0014296, 0.0015061
6: -0.0060265, -0.0053224, -0.0060418, -0.0053514, -0.0006752, 0.0007193
7: -0.0032559, -0.0019368, -0.0031820, -0.0019151, -0.0013408, 0.0012452
8: -0.0043370, -0.0014382, -0.0044025, -0.0015385, -0.0027985, 0.0029644
9: 1.0004292, 1.0008836, 1.0004410, 1.0007743, -0.0003451, 0.0004426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003493, upper bound: 0.0003646
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003205, upper bound: 0.0003253
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0039428, -0.0016366, -0.0024031, 0.0022255
1: 0.0049900, 0.0066135, 0.0049754, 0.0065367, -0.0015466, 0.0016381
2: 0.0105249, 0.0149022, 0.0107018, 0.0150746, -0.0040537, 0.0037062
3: -0.0047243, -0.0027712, -0.0047573, -0.0028598, -0.0018645, 0.0019861
4: 0.0045530, 0.0052054, 0.0045393, 0.0051763, -0.0006115, 0.0006539
5: -0.0023756, -0.0009070, -0.0024557, -0.0009568, -0.0014187, 0.0015487
6: -0.0060265, -0.0053224, -0.0060468, -0.0053530, -0.0006736, 0.0007243
7: -0.0032559, -0.0019368, -0.0032045, -0.0019478, -0.0013081, 0.0012677
8: -0.0043370, -0.0014382, -0.0044575, -0.0015528, -0.0027841, 0.0030193
9: 1.0004292, 1.0008836, 1.0004396, 1.0007663, -0.0003371, 0.0004439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003493, upper bound: 0.0003646
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003205, upper bound: 0.0003253
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0039512, -0.0016674, -0.0023620, 0.0022618
1: 0.0050022, 0.0066194, 0.0049610, 0.0065324, -0.0015302, 0.0016583
2: 0.0105478, 0.0149702, 0.0106815, 0.0149987, -0.0039546, 0.0037952
3: -0.0047174, -0.0027692, -0.0047633, -0.0028611, -0.0018562, 0.0019941
4: 0.0045534, 0.0052053, 0.0045394, 0.0051765, -0.0006122, 0.0006529
5: -0.0024164, -0.0009202, -0.0024131, -0.0009460, -0.0014704, 0.0014929
6: -0.0060307, -0.0053243, -0.0060418, -0.0053514, -0.0006793, 0.0007174
7: -0.0032847, -0.0019655, -0.0031820, -0.0019151, -0.0013696, 0.0012165
8: -0.0043862, -0.0014543, -0.0044025, -0.0015385, -0.0028477, 0.0029482
9: 1.0004270, 1.0008731, 1.0004410, 1.0007743, -0.0003473, 0.0004321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003474, upper bound: 0.0003626
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003100, upper bound: 0.0003083
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0039428, -0.0016366, -0.0023928, 0.0022534
1: 0.0050022, 0.0066194, 0.0049754, 0.0065367, -0.0015345, 0.0016440
2: 0.0105478, 0.0149702, 0.0107018, 0.0150746, -0.0040128, 0.0037586
3: -0.0047174, -0.0027692, -0.0047573, -0.0028598, -0.0018576, 0.0019881
4: 0.0045534, 0.0052053, 0.0045393, 0.0051763, -0.0006177, 0.0006592
5: -0.0024164, -0.0009202, -0.0024557, -0.0009568, -0.0014596, 0.0015355
6: -0.0060307, -0.0053243, -0.0060468, -0.0053530, -0.0006777, 0.0007224
7: -0.0032847, -0.0019655, -0.0032045, -0.0019478, -0.0013368, 0.0012390
8: -0.0043862, -0.0014543, -0.0044575, -0.0015528, -0.0028333, 0.0030032
9: 1.0004270, 1.0008731, 1.0004396, 1.0007663, -0.0003393, 0.0004334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003474, upper bound: 0.0003626
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003100, upper bound: 0.0003083
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0040630, -0.0017093, -0.0023304, 0.0023457
1: 0.0049900, 0.0066135, 0.0049768, 0.0066208, -0.0016308, 0.0016367
2: 0.0105249, 0.0149022, 0.0104779, 0.0149136, -0.0039226, 0.0039481
3: -0.0047243, -0.0027712, -0.0047365, -0.0027597, -0.0019646, 0.0019653
4: 0.0045530, 0.0052054, 0.0045490, 0.0052098, -0.0006427, 0.0006412
5: -0.0023756, -0.0009070, -0.0023801, -0.0008863, -0.0014892, 0.0014731
6: -0.0060265, -0.0053224, -0.0060301, -0.0053163, -0.0007103, 0.0007077
7: -0.0032559, -0.0019368, -0.0032520, -0.0019201, -0.0013358, 0.0013152
8: -0.0043370, -0.0014382, -0.0043444, -0.0014065, -0.0029305, 0.0029063
9: 1.0004292, 1.0008836, 1.0004286, 1.0009116, -0.0004824, 0.0004550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003616, upper bound: 0.0003754
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003341, upper bound: 0.0003358
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0040556, -0.0016762, -0.0023635, 0.0023383
1: 0.0049900, 0.0066135, 0.0049888, 0.0066256, -0.0016356, 0.0016247
2: 0.0105249, 0.0149022, 0.0104947, 0.0149935, -0.0040014, 0.0039297
3: -0.0047243, -0.0027712, -0.0047309, -0.0027582, -0.0019661, 0.0019597
4: 0.0045530, 0.0052054, 0.0045490, 0.0052096, -0.0006426, 0.0006418
5: -0.0023756, -0.0009070, -0.0024220, -0.0008967, -0.0014789, 0.0015150
6: -0.0060265, -0.0053224, -0.0060353, -0.0053176, -0.0007089, 0.0007129
7: -0.0032559, -0.0019368, -0.0032766, -0.0019509, -0.0013051, 0.0013398
8: -0.0043370, -0.0014382, -0.0044012, -0.0014181, -0.0029189, 0.0029631
9: 1.0004292, 1.0008836, 1.0004270, 1.0009035, -0.0004743, 0.0004566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003616, upper bound: 0.0003754
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003341, upper bound: 0.0003358
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0040630, -0.0017093, -0.0023201, 0.0023736
1: 0.0050022, 0.0066194, 0.0049768, 0.0066208, -0.0016186, 0.0016426
2: 0.0105478, 0.0149702, 0.0104779, 0.0149136, -0.0039034, 0.0040231
3: -0.0047174, -0.0027692, -0.0047365, -0.0027597, -0.0019577, 0.0019673
4: 0.0045534, 0.0052053, 0.0045490, 0.0052098, -0.0006433, 0.0006411
5: -0.0024164, -0.0009202, -0.0023801, -0.0008863, -0.0015301, 0.0014599
6: -0.0060307, -0.0053243, -0.0060301, -0.0053163, -0.0007144, 0.0007058
7: -0.0032847, -0.0019655, -0.0032520, -0.0019201, -0.0013645, 0.0012865
8: -0.0043862, -0.0014543, -0.0043444, -0.0014065, -0.0029797, 0.0028902
9: 1.0004270, 1.0008731, 1.0004286, 1.0009116, -0.0004846, 0.0004445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003590, upper bound: 0.0003728
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003170
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0040556, -0.0016762, -0.0023532, 0.0023662
1: 0.0050022, 0.0066194, 0.0049888, 0.0066256, -0.0016235, 0.0016306
2: 0.0105478, 0.0149702, 0.0104947, 0.0149935, -0.0039615, 0.0039837
3: -0.0047174, -0.0027692, -0.0047309, -0.0027582, -0.0019592, 0.0019617
4: 0.0045534, 0.0052053, 0.0045490, 0.0052096, -0.0006486, 0.0006468
5: -0.0024164, -0.0009202, -0.0024220, -0.0008967, -0.0015197, 0.0015018
6: -0.0060307, -0.0053243, -0.0060353, -0.0053176, -0.0007130, 0.0007110
7: -0.0032847, -0.0019655, -0.0032766, -0.0019509, -0.0013338, 0.0013112
8: -0.0043862, -0.0014543, -0.0044012, -0.0014181, -0.0029681, 0.0029469
9: 1.0004270, 1.0008731, 1.0004270, 1.0009035, -0.0004765, 0.0004461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003590, upper bound: 0.0003728
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003170
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0039512, -0.0016674, -0.0024775, 0.0021862
1: 0.0050150, 0.0066928, 0.0049610, 0.0065324, -0.0015174, 0.0017318
2: 0.0103310, 0.0148092, 0.0106815, 0.0149987, -0.0041813, 0.0036443
3: -0.0046896, -0.0026759, -0.0047633, -0.0028611, -0.0018284, 0.0020874
4: 0.0045649, 0.0052372, 0.0045394, 0.0051765, -0.0005962, 0.0006796
5: -0.0023423, -0.0008497, -0.0024131, -0.0009460, -0.0013963, 0.0015634
6: -0.0060133, -0.0052887, -0.0060418, -0.0053514, -0.0006619, 0.0007530
7: -0.0033235, -0.0019437, -0.0031820, -0.0019151, -0.0014084, 0.0012383
8: -0.0042760, -0.0013106, -0.0044025, -0.0015385, -0.0027375, 0.0030920
9: 1.0004183, 1.0010175, 1.0004410, 1.0007743, -0.0003560, 0.0005765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003633, upper bound: 0.0003777
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003286, upper bound: 0.0003368
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0039428, -0.0016366, -0.0025082, 0.0021779
1: 0.0050150, 0.0066928, 0.0049754, 0.0065367, -0.0015217, 0.0017175
2: 0.0103310, 0.0148092, 0.0107018, 0.0150746, -0.0042611, 0.0036303
3: -0.0046896, -0.0026759, -0.0047573, -0.0028598, -0.0018298, 0.0020814
4: 0.0045649, 0.0052372, 0.0045393, 0.0051763, -0.0005960, 0.0006804
5: -0.0023423, -0.0008497, -0.0024557, -0.0009568, -0.0013854, 0.0016060
6: -0.0060133, -0.0052887, -0.0060468, -0.0053530, -0.0006603, 0.0007580
7: -0.0033235, -0.0019437, -0.0032045, -0.0019478, -0.0013757, 0.0012608
8: -0.0042760, -0.0013106, -0.0044575, -0.0015528, -0.0027232, 0.0031469
9: 1.0004183, 1.0010175, 1.0004396, 1.0007663, -0.0003480, 0.0005778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003633, upper bound: 0.0003777
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003286, upper bound: 0.0003368
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0039512, -0.0016674, -0.0024658, 0.0022130
1: 0.0050250, 0.0066997, 0.0049610, 0.0065324, -0.0015073, 0.0017387
2: 0.0103584, 0.0148763, 0.0106815, 0.0149987, -0.0041556, 0.0037093
3: -0.0046833, -0.0026736, -0.0047633, -0.0028611, -0.0018222, 0.0020897
4: 0.0045654, 0.0052370, 0.0045394, 0.0051765, -0.0005965, 0.0006795
5: -0.0023831, -0.0008633, -0.0024131, -0.0009460, -0.0014371, 0.0015498
6: -0.0060173, -0.0052911, -0.0060418, -0.0053514, -0.0006659, 0.0007506
7: -0.0033504, -0.0019719, -0.0031820, -0.0019151, -0.0014353, 0.0012100
8: -0.0043242, -0.0013300, -0.0044025, -0.0015385, -0.0027857, 0.0030725
9: 1.0004160, 1.0010043, 1.0004410, 1.0007743, -0.0003582, 0.0005634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003619, upper bound: 0.0003766
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003218, upper bound: 0.0003250
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0039428, -0.0016366, -0.0024966, 0.0022047
1: 0.0050250, 0.0066997, 0.0049754, 0.0065367, -0.0015116, 0.0017243
2: 0.0103584, 0.0148763, 0.0107018, 0.0150746, -0.0042162, 0.0036761
3: -0.0046833, -0.0026736, -0.0047573, -0.0028598, -0.0018235, 0.0020837
4: 0.0045654, 0.0052370, 0.0045393, 0.0051763, -0.0006008, 0.0006855
5: -0.0023831, -0.0008633, -0.0024557, -0.0009568, -0.0014262, 0.0015924
6: -0.0060173, -0.0052911, -0.0060468, -0.0053530, -0.0006643, 0.0007556
7: -0.0033504, -0.0019719, -0.0032045, -0.0019478, -0.0014026, 0.0012325
8: -0.0043242, -0.0013300, -0.0044575, -0.0015528, -0.0027713, 0.0031275
9: 1.0004160, 1.0010043, 1.0004396, 1.0007663, -0.0003502, 0.0005647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003619, upper bound: 0.0003766
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003218, upper bound: 0.0003250
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0040630, -0.0017093, -0.0024355, 0.0022980
1: 0.0050150, 0.0066928, 0.0049768, 0.0066208, -0.0016058, 0.0017160
2: 0.0103310, 0.0148092, 0.0104779, 0.0149136, -0.0040652, 0.0038113
3: -0.0046896, -0.0026759, -0.0047365, -0.0027597, -0.0019299, 0.0020606
4: 0.0045649, 0.0052372, 0.0045490, 0.0052098, -0.0006230, 0.0006647
5: -0.0023423, -0.0008497, -0.0023801, -0.0008863, -0.0014559, 0.0015304
6: -0.0060133, -0.0052887, -0.0060301, -0.0053163, -0.0006970, 0.0007414
7: -0.0033235, -0.0019437, -0.0032520, -0.0019201, -0.0014034, 0.0013083
8: -0.0042760, -0.0013106, -0.0043444, -0.0014065, -0.0028695, 0.0030339
9: 1.0004183, 1.0010175, 1.0004286, 1.0009116, -0.0004933, 0.0005889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003519, upper bound: 0.0003581
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0040556, -0.0016762, -0.0024687, 0.0022906
1: 0.0050150, 0.0066928, 0.0049888, 0.0066256, -0.0016106, 0.0017041
2: 0.0103310, 0.0148092, 0.0104947, 0.0149935, -0.0041435, 0.0037954
3: -0.0046896, -0.0026759, -0.0047309, -0.0027582, -0.0019314, 0.0020550
4: 0.0045649, 0.0052372, 0.0045490, 0.0052096, -0.0006229, 0.0006655
5: -0.0023423, -0.0008497, -0.0024220, -0.0008967, -0.0014456, 0.0015723
6: -0.0060133, -0.0052887, -0.0060353, -0.0053176, -0.0006956, 0.0007466
7: -0.0033235, -0.0019437, -0.0032766, -0.0019509, -0.0013726, 0.0013330
8: -0.0042760, -0.0013106, -0.0044012, -0.0014181, -0.0028579, 0.0030907
9: 1.0004183, 1.0010175, 1.0004270, 1.0009035, -0.0004852, 0.0005904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003519, upper bound: 0.0003581
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0040630, -0.0017093, -0.0024239, 0.0023248
1: 0.0050250, 0.0066997, 0.0049768, 0.0066208, -0.0015958, 0.0017229
2: 0.0103584, 0.0148763, 0.0104779, 0.0149136, -0.0040445, 0.0038801
3: -0.0046833, -0.0026736, -0.0047365, -0.0027597, -0.0019236, 0.0020629
4: 0.0045654, 0.0052370, 0.0045490, 0.0052098, -0.0006236, 0.0006646
5: -0.0023831, -0.0008633, -0.0023801, -0.0008863, -0.0014967, 0.0015168
6: -0.0060173, -0.0052911, -0.0060301, -0.0053163, -0.0007010, 0.0007390
7: -0.0033504, -0.0019719, -0.0032520, -0.0019201, -0.0014303, 0.0012801
8: -0.0043242, -0.0013300, -0.0043444, -0.0014065, -0.0029177, 0.0030144
9: 1.0004160, 1.0010043, 1.0004286, 1.0009116, -0.0004956, 0.0005758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003446
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0040556, -0.0016762, -0.0024571, 0.0023174
1: 0.0050250, 0.0066997, 0.0049888, 0.0066256, -0.0016006, 0.0017109
2: 0.0103584, 0.0148763, 0.0104947, 0.0149935, -0.0041035, 0.0038460
3: -0.0046833, -0.0026736, -0.0047309, -0.0027582, -0.0019251, 0.0020573
4: 0.0045654, 0.0052370, 0.0045490, 0.0052096, -0.0006290, 0.0006707
5: -0.0023831, -0.0008633, -0.0024220, -0.0008967, -0.0014864, 0.0015587
6: -0.0060173, -0.0052911, -0.0060353, -0.0053176, -0.0006996, 0.0007442
7: -0.0033504, -0.0019719, -0.0032766, -0.0019509, -0.0013995, 0.0013047
8: -0.0043242, -0.0013300, -0.0044012, -0.0014181, -0.0029061, 0.0030712
9: 1.0004160, 1.0010043, 1.0004270, 1.0009035, -0.0004874, 0.0005773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003446
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0040397, -0.0017173, -0.0023225, 0.0023225
1: 0.0049900, 0.0066135, 0.0049900, 0.0066135, -0.0016235, 0.0016235
2: 0.0105249, 0.0149022, 0.0105249, 0.0149022, -0.0038028, 0.0038028
3: -0.0047243, -0.0027712, -0.0047243, -0.0027712, -0.0019531, 0.0019531
4: 0.0045530, 0.0052054, 0.0045530, 0.0052054, -0.0006433, 0.0006433
5: -0.0023756, -0.0009070, -0.0023756, -0.0009070, -0.0014686, 0.0014686
6: -0.0060265, -0.0053224, -0.0060265, -0.0053224, -0.0007041, 0.0007041
7: -0.0032559, -0.0019368, -0.0032559, -0.0019368, -0.0013191, 0.0013191
8: -0.0043370, -0.0014382, -0.0043370, -0.0014382, -0.0028988, 0.0028988
9: 1.0004292, 1.0008836, 1.0004292, 1.0008836, -0.0004544, 0.0004544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003440, upper bound: 0.0003607
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003160, upper bound: 0.0003234
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0040294, -0.0016894, -0.0023504, 0.0023122
1: 0.0049900, 0.0066135, 0.0050022, 0.0066194, -0.0016294, 0.0016113
2: 0.0105249, 0.0149022, 0.0105478, 0.0149702, -0.0038809, 0.0037833
3: -0.0047243, -0.0027712, -0.0047174, -0.0027692, -0.0019551, 0.0019462
4: 0.0045530, 0.0052054, 0.0045534, 0.0052053, -0.0006431, 0.0006441
5: -0.0023756, -0.0009070, -0.0024164, -0.0009202, -0.0014554, 0.0015094
6: -0.0060265, -0.0053224, -0.0060307, -0.0053243, -0.0007022, 0.0007083
7: -0.0032559, -0.0019368, -0.0032847, -0.0019655, -0.0012904, 0.0013478
8: -0.0043370, -0.0014382, -0.0043862, -0.0014543, -0.0028827, 0.0029480
9: 1.0004292, 1.0008836, 1.0004270, 1.0008731, -0.0004439, 0.0004566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003440, upper bound: 0.0003607
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003160, upper bound: 0.0003234
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0040397, -0.0017173, -0.0023122, 0.0023504
1: 0.0050022, 0.0066194, 0.0049900, 0.0066135, -0.0016113, 0.0016294
2: 0.0105478, 0.0149702, 0.0105249, 0.0149022, -0.0037833, 0.0038809
3: -0.0047174, -0.0027692, -0.0047243, -0.0027712, -0.0019462, 0.0019551
4: 0.0045534, 0.0052053, 0.0045530, 0.0052054, -0.0006441, 0.0006431
5: -0.0024164, -0.0009202, -0.0023756, -0.0009070, -0.0015094, 0.0014554
6: -0.0060307, -0.0053243, -0.0060265, -0.0053224, -0.0007083, 0.0007022
7: -0.0032847, -0.0019655, -0.0032559, -0.0019368, -0.0013478, 0.0012904
8: -0.0043862, -0.0014543, -0.0043370, -0.0014382, -0.0029480, 0.0028827
9: 1.0004270, 1.0008731, 1.0004292, 1.0008836, -0.0004566, 0.0004439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003424, upper bound: 0.0003592
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003068, upper bound: 0.0003068
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0040294, -0.0016894, -0.0023401, 0.0023401
1: 0.0050022, 0.0066194, 0.0050022, 0.0066194, -0.0016172, 0.0016172
2: 0.0105478, 0.0149702, 0.0105478, 0.0149702, -0.0038369, 0.0038369
3: -0.0047174, -0.0027692, -0.0047174, -0.0027692, -0.0019482, 0.0019482
4: 0.0045534, 0.0052053, 0.0045534, 0.0052053, -0.0006487, 0.0006487
5: -0.0024164, -0.0009202, -0.0024164, -0.0009202, -0.0014962, 0.0014962
6: -0.0060307, -0.0053243, -0.0060307, -0.0053243, -0.0007063, 0.0007063
7: -0.0032847, -0.0019655, -0.0032847, -0.0019655, -0.0013192, 0.0013192
8: -0.0043862, -0.0014543, -0.0043862, -0.0014543, -0.0029319, 0.0029319
9: 1.0004270, 1.0008731, 1.0004270, 1.0008731, -0.0004461, 0.0004461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003424, upper bound: 0.0003592
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003068, upper bound: 0.0003068
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0041449, -0.0017649, -0.0022748, 0.0024276
1: 0.0049900, 0.0066135, 0.0050150, 0.0066928, -0.0017028, 0.0015985
2: 0.0105249, 0.0149022, 0.0103310, 0.0148092, -0.0037484, 0.0040239
3: -0.0047243, -0.0027712, -0.0046896, -0.0026759, -0.0020484, 0.0019184
4: 0.0045530, 0.0052054, 0.0045649, 0.0052372, -0.0006745, 0.0006320
5: -0.0023756, -0.0009070, -0.0023423, -0.0008497, -0.0015259, 0.0014353
6: -0.0060265, -0.0053224, -0.0060133, -0.0052887, -0.0007378, 0.0006909
7: -0.0032559, -0.0019368, -0.0033235, -0.0019437, -0.0013122, 0.0013867
8: -0.0043370, -0.0014382, -0.0042760, -0.0013106, -0.0030264, 0.0028379
9: 1.0004292, 1.0008836, 1.0004183, 1.0010175, -0.0005883, 0.0004653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003614, upper bound: 0.0003747
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003340, upper bound: 0.0003356
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0017173, -0.0041332, -0.0017381, -0.0023016, 0.0024160
1: 0.0049900, 0.0066135, 0.0050250, 0.0066997, -0.0017097, 0.0015885
2: 0.0105249, 0.0149022, 0.0103584, 0.0148763, -0.0038187, 0.0040014
3: -0.0047243, -0.0027712, -0.0046833, -0.0026736, -0.0020507, 0.0019121
4: 0.0045530, 0.0052054, 0.0045654, 0.0052370, -0.0006743, 0.0006328
5: -0.0023756, -0.0009070, -0.0023831, -0.0008633, -0.0015123, 0.0014761
6: -0.0060265, -0.0053224, -0.0060173, -0.0052911, -0.0007354, 0.0006948
7: -0.0032559, -0.0019368, -0.0033504, -0.0019719, -0.0012840, 0.0014136
8: -0.0043370, -0.0014382, -0.0043242, -0.0013300, -0.0030070, 0.0028860
9: 1.0004292, 1.0008836, 1.0004160, 1.0010043, -0.0005752, 0.0004675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003614, upper bound: 0.0003747
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003340, upper bound: 0.0003356
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0041449, -0.0017649, -0.0022645, 0.0024555
1: 0.0050022, 0.0066194, 0.0050150, 0.0066928, -0.0016907, 0.0016044
2: 0.0105478, 0.0149702, 0.0103310, 0.0148092, -0.0037290, 0.0041020
3: -0.0047174, -0.0027692, -0.0046896, -0.0026759, -0.0020415, 0.0019204
4: 0.0045534, 0.0052053, 0.0045649, 0.0052372, -0.0006753, 0.0006319
5: -0.0024164, -0.0009202, -0.0023423, -0.0008497, -0.0015667, 0.0014221
6: -0.0060307, -0.0053243, -0.0060133, -0.0052887, -0.0007420, 0.0006889
7: -0.0032847, -0.0019655, -0.0033235, -0.0019437, -0.0013410, 0.0013580
8: -0.0043862, -0.0014543, -0.0042760, -0.0013106, -0.0030756, 0.0028217
9: 1.0004270, 1.0008731, 1.0004183, 1.0010175, -0.0005904, 0.0004548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003588, upper bound: 0.0003724
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003169
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040294, -0.0016894, -0.0041332, -0.0017381, -0.0022913, 0.0024439
1: 0.0050022, 0.0066194, 0.0050250, 0.0066997, -0.0016975, 0.0015944
2: 0.0105478, 0.0149702, 0.0103584, 0.0148763, -0.0037788, 0.0040570
3: -0.0047174, -0.0027692, -0.0046833, -0.0026736, -0.0020438, 0.0019141
4: 0.0045534, 0.0052053, 0.0045654, 0.0052370, -0.0006793, 0.0006374
5: -0.0024164, -0.0009202, -0.0023831, -0.0008633, -0.0015531, 0.0014629
6: -0.0060307, -0.0053243, -0.0060173, -0.0052911, -0.0007396, 0.0006929
7: -0.0032847, -0.0019655, -0.0033504, -0.0019719, -0.0013127, 0.0013849
8: -0.0043862, -0.0014543, -0.0043242, -0.0013300, -0.0030561, 0.0028699
9: 1.0004270, 1.0008731, 1.0004160, 1.0010043, -0.0005773, 0.0004570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003588, upper bound: 0.0003724
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003169
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0040397, -0.0017173, -0.0024276, 0.0022748
1: 0.0050150, 0.0066928, 0.0049900, 0.0066135, -0.0015985, 0.0017028
2: 0.0103310, 0.0148092, 0.0105249, 0.0149022, -0.0040239, 0.0037484
3: -0.0046896, -0.0026759, -0.0047243, -0.0027712, -0.0019184, 0.0020484
4: 0.0045649, 0.0052372, 0.0045530, 0.0052054, -0.0006320, 0.0006745
5: -0.0023423, -0.0008497, -0.0023756, -0.0009070, -0.0014353, 0.0015259
6: -0.0060133, -0.0052887, -0.0060265, -0.0053224, -0.0006909, 0.0007378
7: -0.0033235, -0.0019437, -0.0032559, -0.0019368, -0.0013867, 0.0013122
8: -0.0042760, -0.0013106, -0.0043370, -0.0014382, -0.0028379, 0.0030264
9: 1.0004183, 1.0010175, 1.0004292, 1.0008836, -0.0004653, 0.0005883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0040294, -0.0016894, -0.0024555, 0.0022645
1: 0.0050150, 0.0066928, 0.0050022, 0.0066194, -0.0016044, 0.0016907
2: 0.0103310, 0.0148092, 0.0105478, 0.0149702, -0.0041020, 0.0037290
3: -0.0046896, -0.0026759, -0.0047174, -0.0027692, -0.0019204, 0.0020415
4: 0.0045649, 0.0052372, 0.0045534, 0.0052053, -0.0006319, 0.0006753
5: -0.0023423, -0.0008497, -0.0024164, -0.0009202, -0.0014221, 0.0015667
6: -0.0060133, -0.0052887, -0.0060307, -0.0053243, -0.0006889, 0.0007420
7: -0.0033235, -0.0019437, -0.0032847, -0.0019655, -0.0013580, 0.0013410
8: -0.0042760, -0.0013106, -0.0043862, -0.0014543, -0.0028217, 0.0030756
9: 1.0004183, 1.0010175, 1.0004270, 1.0008731, -0.0004548, 0.0005904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0040397, -0.0017173, -0.0024160, 0.0023016
1: 0.0050250, 0.0066997, 0.0049900, 0.0066135, -0.0015885, 0.0017097
2: 0.0103584, 0.0148763, 0.0105249, 0.0149022, -0.0040014, 0.0038187
3: -0.0046833, -0.0026736, -0.0047243, -0.0027712, -0.0019121, 0.0020507
4: 0.0045654, 0.0052370, 0.0045530, 0.0052054, -0.0006328, 0.0006743
5: -0.0023831, -0.0008633, -0.0023756, -0.0009070, -0.0014761, 0.0015123
6: -0.0060173, -0.0052911, -0.0060265, -0.0053224, -0.0006948, 0.0007354
7: -0.0033504, -0.0019719, -0.0032559, -0.0019368, -0.0014136, 0.0012840
8: -0.0043242, -0.0013300, -0.0043370, -0.0014382, -0.0028860, 0.0030070
9: 1.0004160, 1.0010043, 1.0004292, 1.0008836, -0.0004675, 0.0005752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003168, upper bound: 0.0003209
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0040294, -0.0016894, -0.0024439, 0.0022913
1: 0.0050250, 0.0066997, 0.0050022, 0.0066194, -0.0015944, 0.0016975
2: 0.0103584, 0.0148763, 0.0105478, 0.0149702, -0.0040570, 0.0037788
3: -0.0046833, -0.0026736, -0.0047174, -0.0027692, -0.0019141, 0.0020438
4: 0.0045654, 0.0052370, 0.0045534, 0.0052053, -0.0006374, 0.0006793
5: -0.0023831, -0.0008633, -0.0024164, -0.0009202, -0.0014629, 0.0015531
6: -0.0060173, -0.0052911, -0.0060307, -0.0053243, -0.0006929, 0.0007396
7: -0.0033504, -0.0019719, -0.0032847, -0.0019655, -0.0013849, 0.0013127
8: -0.0043242, -0.0013300, -0.0043862, -0.0014543, -0.0028699, 0.0030561
9: 1.0004160, 1.0010043, 1.0004270, 1.0008731, -0.0004570, 0.0005773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003168, upper bound: 0.0003209
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0041449, -0.0017649, -0.0023800, 0.0023800
1: 0.0050150, 0.0066928, 0.0050150, 0.0066928, -0.0016779, 0.0016779
2: 0.0103310, 0.0148092, 0.0103310, 0.0148092, -0.0038923, 0.0038923
3: -0.0046896, -0.0026759, -0.0046896, -0.0026759, -0.0020137, 0.0020137
4: 0.0045649, 0.0052372, 0.0045649, 0.0052372, -0.0006502, 0.0006502
5: -0.0023423, -0.0008497, -0.0023423, -0.0008497, -0.0014925, 0.0014925
6: -0.0060133, -0.0052887, -0.0060133, -0.0052887, -0.0007245, 0.0007245
7: -0.0033235, -0.0019437, -0.0033235, -0.0019437, -0.0013798, 0.0013798
8: -0.0042760, -0.0013106, -0.0042760, -0.0013106, -0.0029655, 0.0029655
9: 1.0004183, 1.0010175, 1.0004183, 1.0010175, -0.0005991, 0.0005991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003578
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041449, -0.0017649, -0.0041332, -0.0017381, -0.0024068, 0.0023683
1: 0.0050150, 0.0066928, 0.0050250, 0.0066997, -0.0016847, 0.0016678
2: 0.0103310, 0.0148092, 0.0103584, 0.0148763, -0.0039660, 0.0038714
3: -0.0046896, -0.0026759, -0.0046833, -0.0026736, -0.0020159, 0.0020074
4: 0.0045649, 0.0052372, 0.0045654, 0.0052370, -0.0006500, 0.0006510
5: -0.0023423, -0.0008497, -0.0023831, -0.0008633, -0.0014790, 0.0015334
6: -0.0060133, -0.0052887, -0.0060173, -0.0052911, -0.0007222, 0.0007285
7: -0.0033235, -0.0019437, -0.0033504, -0.0019719, -0.0013516, 0.0014067
8: -0.0042760, -0.0013106, -0.0043242, -0.0013300, -0.0029460, 0.0030136
9: 1.0004183, 1.0010175, 1.0004160, 1.0010043, -0.0005860, 0.0006014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003578
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0041449, -0.0017649, -0.0023683, 0.0024068
1: 0.0050250, 0.0066997, 0.0050150, 0.0066928, -0.0016678, 0.0016847
2: 0.0103584, 0.0148763, 0.0103310, 0.0148092, -0.0038714, 0.0039660
3: -0.0046833, -0.0026736, -0.0046896, -0.0026759, -0.0020074, 0.0020159
4: 0.0045654, 0.0052370, 0.0045649, 0.0052372, -0.0006510, 0.0006500
5: -0.0023831, -0.0008633, -0.0023423, -0.0008497, -0.0015334, 0.0014790
6: -0.0060173, -0.0052911, -0.0060133, -0.0052887, -0.0007285, 0.0007222
7: -0.0033504, -0.0019719, -0.0033235, -0.0019437, -0.0014067, 0.0013516
8: -0.0043242, -0.0013300, -0.0042760, -0.0013106, -0.0030136, 0.0029460
9: 1.0004160, 1.0010043, 1.0004183, 1.0010175, -0.0006014, 0.0005860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003442
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041332, -0.0017381, -0.0041332, -0.0017381, -0.0023951, 0.0023951
1: 0.0050250, 0.0066997, 0.0050250, 0.0066997, -0.0016747, 0.0016747
2: 0.0103584, 0.0148763, 0.0103584, 0.0148763, -0.0039224, 0.0039224
3: -0.0046833, -0.0026736, -0.0046833, -0.0026736, -0.0020097, 0.0020097
4: 0.0045654, 0.0052370, 0.0045654, 0.0052370, -0.0006557, 0.0006557
5: -0.0023831, -0.0008633, -0.0023831, -0.0008633, -0.0015198, 0.0015198
6: -0.0060173, -0.0052911, -0.0060173, -0.0052911, -0.0007261, 0.0007261
7: -0.0033504, -0.0019719, -0.0033504, -0.0019719, -0.0013785, 0.0013785
8: -0.0043242, -0.0013300, -0.0043242, -0.0013300, -0.0029941, 0.0029941
9: 1.0004160, 1.0010043, 1.0004160, 1.0010043, -0.0005883, 0.0005883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003442
time: 0.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.76 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004688, upper bound: 0.0004702
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004688, upper bound: 0.0004702
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004673, upper bound: 0.0004676
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004673, upper bound: 0.0004676
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004784
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004760
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004784
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004760
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004757
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004848, upper bound: 0.0004737
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004757
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004848, upper bound: 0.0004737
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003636, upper bound: 0.0003778
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003293, upper bound: 0.0003380
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003636, upper bound: 0.0003778
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003293, upper bound: 0.0003380
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003621, upper bound: 0.0003766
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003226, upper bound: 0.0003253
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003621, upper bound: 0.0003766
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003226, upper bound: 0.0003253
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003520, upper bound: 0.0003586
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003520, upper bound: 0.0003586
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003449, upper bound: 0.0003448
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003449, upper bound: 0.0003448
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004682, upper bound: 0.0004696
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004658, upper bound: 0.0004672
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004682, upper bound: 0.0004696
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004658, upper bound: 0.0004672
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004666, upper bound: 0.0004670
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004650, upper bound: 0.0004652
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004666, upper bound: 0.0004670
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004650, upper bound: 0.0004652
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004780
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004756
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004904, upper bound: 0.0004780
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004862, upper bound: 0.0004756
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004754
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004734
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004879, upper bound: 0.0004754
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0004847, upper bound: 0.0004734
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003170, upper bound: 0.0003209
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003170, upper bound: 0.0003209
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003580
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003580
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003447, upper bound: 0.0003442
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003447, upper bound: 0.0003442
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003493, upper bound: 0.0003646
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003205, upper bound: 0.0003253
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003493, upper bound: 0.0003646
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003205, upper bound: 0.0003253
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003474, upper bound: 0.0003626
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003100, upper bound: 0.0003083
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003474, upper bound: 0.0003626
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003100, upper bound: 0.0003083
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003616, upper bound: 0.0003754
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003341, upper bound: 0.0003358
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003616, upper bound: 0.0003754
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003341, upper bound: 0.0003358
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003590, upper bound: 0.0003728
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003170
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003590, upper bound: 0.0003728
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003170
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003633, upper bound: 0.0003777
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003286, upper bound: 0.0003368
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003633, upper bound: 0.0003777
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003286, upper bound: 0.0003368
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003619, upper bound: 0.0003766
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003218, upper bound: 0.0003250
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003619, upper bound: 0.0003766
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003218, upper bound: 0.0003250
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003519, upper bound: 0.0003581
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003881, upper bound: 0.0004019
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003519, upper bound: 0.0003581
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003446
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003865, upper bound: 0.0004009
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003446
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003440, upper bound: 0.0003607
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003160, upper bound: 0.0003234
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003440, upper bound: 0.0003607
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003160, upper bound: 0.0003234
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003424, upper bound: 0.0003592
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003068, upper bound: 0.0003068
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003424, upper bound: 0.0003592
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003068, upper bound: 0.0003068
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003614, upper bound: 0.0003747
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003340, upper bound: 0.0003356
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003614, upper bound: 0.0003747
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003340, upper bound: 0.0003356
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003588, upper bound: 0.0003724
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003169
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003588, upper bound: 0.0003724
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003209, upper bound: 0.0003169
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003577, upper bound: 0.0003730
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003235, upper bound: 0.0003325
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003168, upper bound: 0.0003209
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003561, upper bound: 0.0003720
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003168, upper bound: 0.0003209
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003578
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003876, upper bound: 0.0004008
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003516, upper bound: 0.0003578
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003442
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003860, upper bound: 0.0003999
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 9, lower bound: -0.0003444, upper bound: 0.0003442

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0039512, -0.0016674, -0.0022838, 0.0022649
1: 0.0049805, 0.0065324, 0.0049610, 0.0065324, -0.0015519, 0.0015713
2: 0.0106823, 0.0149661, 0.0106815, 0.0149987, -0.0037415, 0.0036207
3: -0.0047422, -0.0028611, -0.0047633, -0.0028611, -0.0018810, 0.0019021
4: 0.0045460, 0.0051765, 0.0045394, 0.0051765, -0.0004218, 0.0005951
5: -0.0024038, -0.0009481, -0.0024131, -0.0009460, -0.0014578, 0.0014650
6: -0.0060353, -0.0053514, -0.0060418, -0.0053514, -0.0006839, 0.0006904
7: -0.0031820, -0.0019364, -0.0031820, -0.0019151, -0.0012669, 0.0012455
8: -0.0043820, -0.0015394, -0.0044025, -0.0015385, -0.0028435, 0.0028631
9: 1.0004410, 1.0007735, 1.0004410, 1.0007743, -0.0003333, 0.0003326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0039512, -0.0016727, -0.0022788, 0.0022668
1: 0.0049849, 0.0065360, 0.0049667, 0.0065324, -0.0015475, 0.0015693
2: 0.0106827, 0.0149735, 0.0106817, 0.0149897, -0.0037224, 0.0036265
3: -0.0047382, -0.0028582, -0.0047571, -0.0028611, -0.0018771, 0.0018989
4: 0.0045470, 0.0051773, 0.0045413, 0.0051765, -0.0004219, 0.0005893
5: -0.0024098, -0.0009491, -0.0024107, -0.0009464, -0.0014633, 0.0014616
6: -0.0060351, -0.0053509, -0.0060399, -0.0053514, -0.0006837, 0.0006890
7: -0.0031902, -0.0019424, -0.0031820, -0.0019210, -0.0012692, 0.0012396
8: -0.0043876, -0.0015398, -0.0043972, -0.0015387, -0.0028489, 0.0028574
9: 1.0004402, 1.0007747, 1.0004410, 1.0007741, -0.0003339, 0.0003338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0039428, -0.0016366, -0.0023145, 0.0022566
1: 0.0049805, 0.0065324, 0.0049754, 0.0065367, -0.0015562, 0.0015570
2: 0.0106823, 0.0149661, 0.0107018, 0.0150746, -0.0038213, 0.0036052
3: -0.0047422, -0.0028611, -0.0047573, -0.0028598, -0.0018824, 0.0018961
4: 0.0045460, 0.0051765, 0.0045393, 0.0051763, -0.0004215, 0.0005959
5: -0.0024038, -0.0009481, -0.0024557, -0.0009568, -0.0014469, 0.0015076
6: -0.0060353, -0.0053514, -0.0060468, -0.0053530, -0.0006823, 0.0006954
7: -0.0031820, -0.0019364, -0.0032045, -0.0019478, -0.0012341, 0.0012680
8: -0.0043820, -0.0015394, -0.0044575, -0.0015528, -0.0028291, 0.0029181
9: 1.0004410, 1.0007735, 1.0004396, 1.0007663, -0.0003253, 0.0003339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0039428, -0.0016413, -0.0023102, 0.0022584
1: 0.0049849, 0.0065360, 0.0049807, 0.0065367, -0.0015518, 0.0015553
2: 0.0106827, 0.0149735, 0.0107020, 0.0150674, -0.0038023, 0.0036109
3: -0.0047382, -0.0028582, -0.0047515, -0.0028598, -0.0018784, 0.0018933
4: 0.0045470, 0.0051773, 0.0045411, 0.0051763, -0.0004216, 0.0005901
5: -0.0024098, -0.0009491, -0.0024535, -0.0009572, -0.0014525, 0.0015045
6: -0.0060351, -0.0053509, -0.0060451, -0.0053530, -0.0006821, 0.0006942
7: -0.0031902, -0.0019424, -0.0032045, -0.0019535, -0.0012366, 0.0012621
8: -0.0043876, -0.0015398, -0.0044527, -0.0015530, -0.0028346, 0.0029129
9: 1.0004402, 1.0007747, 1.0004396, 1.0007660, -0.0003258, 0.0003351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0039512, -0.0016674, -0.0022754, 0.0023000
1: 0.0049918, 0.0065367, 0.0049610, 0.0065324, -0.0015406, 0.0015756
2: 0.0107026, 0.0150499, 0.0106815, 0.0149987, -0.0037263, 0.0037067
3: -0.0047398, -0.0028598, -0.0047633, -0.0028611, -0.0018787, 0.0019035
4: 0.0045448, 0.0051763, 0.0045394, 0.0051765, -0.0004229, 0.0005949
5: -0.0024483, -0.0009590, -0.0024131, -0.0009460, -0.0015023, 0.0014541
6: -0.0060416, -0.0053530, -0.0060418, -0.0053514, -0.0006903, 0.0006888
7: -0.0032045, -0.0019668, -0.0031820, -0.0019151, -0.0012894, 0.0012151
8: -0.0044411, -0.0015538, -0.0044025, -0.0015385, -0.0029027, 0.0028487
9: 1.0004396, 1.0007656, 1.0004410, 1.0007743, -0.0003346, 0.0003246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0039512, -0.0016727, -0.0022704, 0.0023022
1: 0.0049959, 0.0065403, 0.0049667, 0.0065324, -0.0015364, 0.0015736
2: 0.0107029, 0.0150580, 0.0106817, 0.0149897, -0.0037073, 0.0037138
3: -0.0047360, -0.0028568, -0.0047571, -0.0028611, -0.0018748, 0.0019003
4: 0.0045458, 0.0051772, 0.0045413, 0.0051765, -0.0004230, 0.0005891
5: -0.0024551, -0.0009597, -0.0024107, -0.0009464, -0.0015087, 0.0014510
6: -0.0060415, -0.0053525, -0.0060399, -0.0053514, -0.0006901, 0.0006874
7: -0.0032140, -0.0019724, -0.0031820, -0.0019210, -0.0012931, 0.0012096
8: -0.0044474, -0.0015542, -0.0043972, -0.0015387, -0.0029088, 0.0028430
9: 1.0004387, 1.0007666, 1.0004410, 1.0007741, -0.0003355, 0.0003257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0039428, -0.0016366, -0.0023061, 0.0022916
1: 0.0049918, 0.0065367, 0.0049754, 0.0065367, -0.0015449, 0.0015613
2: 0.0107026, 0.0150499, 0.0107018, 0.0150746, -0.0037824, 0.0036640
3: -0.0047398, -0.0028598, -0.0047573, -0.0028598, -0.0018800, 0.0018975
4: 0.0045448, 0.0051763, 0.0045393, 0.0051763, -0.0004226, 0.0006005
5: -0.0024483, -0.0009590, -0.0024557, -0.0009568, -0.0014915, 0.0014967
6: -0.0060416, -0.0053530, -0.0060468, -0.0053530, -0.0006887, 0.0006938
7: -0.0032045, -0.0019668, -0.0032045, -0.0019478, -0.0012567, 0.0012377
8: -0.0044411, -0.0015538, -0.0044575, -0.0015528, -0.0028883, 0.0029037
9: 1.0004396, 1.0007656, 1.0004396, 1.0007663, -0.0003266, 0.0003259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0039428, -0.0016413, -0.0023018, 0.0022938
1: 0.0049959, 0.0065403, 0.0049807, 0.0065367, -0.0015407, 0.0015596
2: 0.0107029, 0.0150580, 0.0107020, 0.0150674, -0.0037635, 0.0036706
3: -0.0047360, -0.0028568, -0.0047515, -0.0028598, -0.0018762, 0.0018947
4: 0.0045458, 0.0051772, 0.0045411, 0.0051763, -0.0004228, 0.0005944
5: -0.0024551, -0.0009597, -0.0024535, -0.0009572, -0.0014979, 0.0014938
6: -0.0060415, -0.0053525, -0.0060451, -0.0053530, -0.0006885, 0.0006926
7: -0.0032140, -0.0019724, -0.0032045, -0.0019535, -0.0012605, 0.0012321
8: -0.0044474, -0.0015542, -0.0044527, -0.0015530, -0.0028944, 0.0028985
9: 1.0004387, 1.0007666, 1.0004396, 1.0007660, -0.0003273, 0.0003270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0040630, -0.0017093, -0.0022418, 0.0023767
1: 0.0049805, 0.0065324, 0.0049768, 0.0066208, -0.0016403, 0.0015556
2: 0.0106823, 0.0149661, 0.0104779, 0.0149136, -0.0036902, 0.0038582
3: -0.0047422, -0.0028611, -0.0047365, -0.0027597, -0.0019825, 0.0018754
4: 0.0045460, 0.0051765, 0.0045490, 0.0052098, -0.0004627, 0.0005832
5: -0.0024038, -0.0009481, -0.0023801, -0.0008863, -0.0015174, 0.0014320
6: -0.0060353, -0.0053514, -0.0060301, -0.0053163, -0.0007190, 0.0006788
7: -0.0031820, -0.0019364, -0.0032520, -0.0019201, -0.0012619, 0.0013156
8: -0.0043820, -0.0015394, -0.0043444, -0.0014065, -0.0029755, 0.0028050
9: 1.0004410, 1.0007735, 1.0004286, 1.0009116, -0.0004706, 0.0003450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003991, upper bound: 0.0003856
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003428, upper bound: 0.0003428
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0040630, -0.0017144, -0.0022370, 0.0023786
1: 0.0049849, 0.0065360, 0.0049822, 0.0066208, -0.0016359, 0.0015538
2: 0.0106827, 0.0149735, 0.0104781, 0.0149053, -0.0036710, 0.0038639
3: -0.0047382, -0.0028582, -0.0047305, -0.0027597, -0.0019785, 0.0018723
4: 0.0045470, 0.0051773, 0.0045510, 0.0052098, -0.0004627, 0.0005773
5: -0.0024098, -0.0009491, -0.0023776, -0.0008868, -0.0015230, 0.0014285
6: -0.0060351, -0.0053509, -0.0060283, -0.0053163, -0.0007188, 0.0006774
7: -0.0031902, -0.0019424, -0.0032520, -0.0019260, -0.0012641, 0.0013096
8: -0.0043876, -0.0015398, -0.0043390, -0.0014067, -0.0029809, 0.0027991
9: 1.0004402, 1.0007747, 1.0004286, 1.0009116, -0.0004714, 0.0003462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003785, upper bound: 0.0003644
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003315, upper bound: 0.0003278
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0040556, -0.0016762, -0.0022750, 0.0023693
1: 0.0049805, 0.0065324, 0.0049888, 0.0066256, -0.0016452, 0.0015436
2: 0.0106823, 0.0149661, 0.0104947, 0.0149935, -0.0037690, 0.0038383
3: -0.0047422, -0.0028611, -0.0047309, -0.0027582, -0.0019840, 0.0018698
4: 0.0045460, 0.0051765, 0.0045490, 0.0052096, -0.0004624, 0.0005839
5: -0.0024038, -0.0009481, -0.0024220, -0.0008967, -0.0015071, 0.0014739
6: -0.0060353, -0.0053514, -0.0060353, -0.0053176, -0.0007176, 0.0006840
7: -0.0031820, -0.0019364, -0.0032766, -0.0019509, -0.0012311, 0.0013402
8: -0.0043820, -0.0015394, -0.0044012, -0.0014181, -0.0029639, 0.0028618
9: 1.0004410, 1.0007735, 1.0004270, 1.0009035, -0.0004625, 0.0003465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003930, upper bound: 0.0003806
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003182, upper bound: 0.0003224
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0040556, -0.0016806, -0.0022709, 0.0023712
1: 0.0049849, 0.0065360, 0.0049938, 0.0066256, -0.0016407, 0.0015422
2: 0.0106827, 0.0149735, 0.0104949, 0.0149860, -0.0037499, 0.0038440
3: -0.0047382, -0.0028582, -0.0047253, -0.0027582, -0.0019800, 0.0018671
4: 0.0045470, 0.0051773, 0.0045508, 0.0052096, -0.0004625, 0.0005779
5: -0.0024098, -0.0009491, -0.0024198, -0.0008971, -0.0015126, 0.0014707
6: -0.0060351, -0.0053509, -0.0060337, -0.0053176, -0.0007174, 0.0006828
7: -0.0031902, -0.0019424, -0.0032766, -0.0019565, -0.0012337, 0.0013342
8: -0.0043876, -0.0015398, -0.0043963, -0.0014183, -0.0029693, 0.0028564
9: 1.0004402, 1.0007747, 1.0004270, 1.0009034, -0.0004631, 0.0003477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003725, upper bound: 0.0003589
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003062, upper bound: 0.0003079
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0040630, -0.0017093, -0.0022334, 0.0024118
1: 0.0049918, 0.0065367, 0.0049768, 0.0066208, -0.0016290, 0.0015598
2: 0.0107026, 0.0150499, 0.0104779, 0.0149136, -0.0036750, 0.0039442
3: -0.0047398, -0.0028598, -0.0047365, -0.0027597, -0.0019801, 0.0018767
4: 0.0045448, 0.0051763, 0.0045490, 0.0052098, -0.0004638, 0.0005831
5: -0.0024483, -0.0009590, -0.0023801, -0.0008863, -0.0015620, 0.0014210
6: -0.0060416, -0.0053530, -0.0060301, -0.0053163, -0.0007254, 0.0006772
7: -0.0032045, -0.0019668, -0.0032520, -0.0019201, -0.0012844, 0.0012852
8: -0.0044411, -0.0015538, -0.0043444, -0.0014065, -0.0030347, 0.0027906
9: 1.0004396, 1.0007656, 1.0004286, 1.0009116, -0.0004719, 0.0003370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003702, upper bound: 0.0003535
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003173, upper bound: 0.0003126
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0040630, -0.0017144, -0.0022286, 0.0024140
1: 0.0049959, 0.0065403, 0.0049822, 0.0066208, -0.0016249, 0.0015582
2: 0.0107029, 0.0150580, 0.0104781, 0.0149053, -0.0036559, 0.0039512
3: -0.0047360, -0.0028568, -0.0047305, -0.0027597, -0.0019763, 0.0018737
4: 0.0045458, 0.0051772, 0.0045510, 0.0052098, -0.0004639, 0.0005772
5: -0.0024551, -0.0009597, -0.0023776, -0.0008868, -0.0015683, 0.0014179
6: -0.0060415, -0.0053525, -0.0060283, -0.0053163, -0.0007252, 0.0006757
7: -0.0032140, -0.0019724, -0.0032520, -0.0019260, -0.0012880, 0.0012797
8: -0.0044474, -0.0015542, -0.0043390, -0.0014067, -0.0030407, 0.0027848
9: 1.0004387, 1.0007666, 1.0004286, 1.0009116, -0.0004729, 0.0003381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003487, upper bound: 0.0003311
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003036, upper bound: 0.0002952
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0040556, -0.0016762, -0.0022666, 0.0024044
1: 0.0049918, 0.0065367, 0.0049888, 0.0066256, -0.0016338, 0.0015479
2: 0.0107026, 0.0150499, 0.0104947, 0.0149935, -0.0037311, 0.0039007
3: -0.0047398, -0.0028598, -0.0047309, -0.0027582, -0.0019816, 0.0018711
4: 0.0045448, 0.0051763, 0.0045490, 0.0052096, -0.0004635, 0.0005881
5: -0.0024483, -0.0009590, -0.0024220, -0.0008967, -0.0015516, 0.0014630
6: -0.0060416, -0.0053530, -0.0060353, -0.0053176, -0.0007240, 0.0006824
7: -0.0032045, -0.0019668, -0.0032766, -0.0019509, -0.0012536, 0.0013098
8: -0.0044411, -0.0015538, -0.0044012, -0.0014181, -0.0030230, 0.0028474
9: 1.0004396, 1.0007656, 1.0004270, 1.0009035, -0.0004638, 0.0003386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003699, upper bound: 0.0003534
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003071, upper bound: 0.0003067
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0040556, -0.0016806, -0.0022625, 0.0024066
1: 0.0049959, 0.0065403, 0.0049938, 0.0066256, -0.0016297, 0.0015465
2: 0.0107029, 0.0150580, 0.0104949, 0.0149860, -0.0037123, 0.0039073
3: -0.0047360, -0.0028568, -0.0047253, -0.0027582, -0.0019778, 0.0018685
4: 0.0045458, 0.0051772, 0.0045508, 0.0052096, -0.0004637, 0.0005818
5: -0.0024551, -0.0009597, -0.0024198, -0.0008971, -0.0015580, 0.0014601
6: -0.0060415, -0.0053525, -0.0060337, -0.0053176, -0.0007238, 0.0006811
7: -0.0032140, -0.0019724, -0.0032766, -0.0019565, -0.0012576, 0.0013043
8: -0.0044474, -0.0015542, -0.0043963, -0.0014183, -0.0030291, 0.0028421
9: 1.0004387, 1.0007666, 1.0004270, 1.0009034, -0.0004647, 0.0003396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003478, upper bound: 0.0003305
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002947, upper bound: 0.0002901
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0040397, -0.0017173, -0.0022339, 0.0023535
1: 0.0049805, 0.0065324, 0.0049900, 0.0066135, -0.0016331, 0.0015423
2: 0.0106823, 0.0149661, 0.0105249, 0.0149022, -0.0036830, 0.0038402
3: -0.0047422, -0.0028611, -0.0047243, -0.0027712, -0.0019710, 0.0018632
4: 0.0045460, 0.0051765, 0.0045530, 0.0052054, -0.0004979, 0.0005913
5: -0.0024038, -0.0009481, -0.0023756, -0.0009070, -0.0014968, 0.0014275
6: -0.0060353, -0.0053514, -0.0060265, -0.0053224, -0.0007128, 0.0006752
7: -0.0031820, -0.0019364, -0.0032559, -0.0019368, -0.0012452, 0.0013195
8: -0.0043820, -0.0015394, -0.0043370, -0.0014382, -0.0029438, 0.0027976
9: 1.0004410, 1.0007735, 1.0004292, 1.0008836, -0.0004426, 0.0003444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003831, upper bound: 0.0003706
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003232, upper bound: 0.0003273
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0040397, -0.0017233, -0.0022282, 0.0023554
1: 0.0049849, 0.0065360, 0.0049960, 0.0066135, -0.0016286, 0.0015400
2: 0.0106827, 0.0149735, 0.0105251, 0.0148918, -0.0036632, 0.0038458
3: -0.0047382, -0.0028582, -0.0047176, -0.0027712, -0.0019670, 0.0018595
4: 0.0045470, 0.0051773, 0.0045551, 0.0052054, -0.0004980, 0.0005852
5: -0.0024098, -0.0009491, -0.0023729, -0.0009075, -0.0015023, 0.0014238
6: -0.0060351, -0.0053509, -0.0060245, -0.0053224, -0.0007126, 0.0006736
7: -0.0031902, -0.0019424, -0.0032559, -0.0019429, -0.0012472, 0.0013135
8: -0.0043876, -0.0015398, -0.0043308, -0.0014384, -0.0029492, 0.0027910
9: 1.0004402, 1.0007747, 1.0004292, 1.0008835, -0.0004432, 0.0003456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003639, upper bound: 0.0003496
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003147, upper bound: 0.0003156
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0040294, -0.0016894, -0.0022618, 0.0023432
1: 0.0049805, 0.0065324, 0.0050022, 0.0066194, -0.0016389, 0.0015302
2: 0.0106823, 0.0149661, 0.0105478, 0.0149702, -0.0037580, 0.0038163
3: -0.0047422, -0.0028611, -0.0047174, -0.0027692, -0.0019730, 0.0018562
4: 0.0045460, 0.0051765, 0.0045534, 0.0052053, -0.0004976, 0.0005918
5: -0.0024038, -0.0009481, -0.0024164, -0.0009202, -0.0014836, 0.0014683
6: -0.0060353, -0.0053514, -0.0060307, -0.0053243, -0.0007109, 0.0006793
7: -0.0031820, -0.0019364, -0.0032847, -0.0019655, -0.0012165, 0.0013482
8: -0.0043820, -0.0015394, -0.0043862, -0.0014543, -0.0029277, 0.0028467
9: 1.0004410, 1.0007735, 1.0004270, 1.0008731, -0.0004321, 0.0003465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003734, upper bound: 0.0003629
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002817, upper bound: 0.0002919
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0040294, -0.0016947, -0.0022568, 0.0023451
1: 0.0049849, 0.0065360, 0.0050078, 0.0066194, -0.0016345, 0.0015282
2: 0.0106827, 0.0149735, 0.0105480, 0.0149617, -0.0037387, 0.0038220
3: -0.0047382, -0.0028582, -0.0047112, -0.0027692, -0.0019690, 0.0018530
4: 0.0045470, 0.0051773, 0.0045554, 0.0052053, -0.0004976, 0.0005857
5: -0.0024098, -0.0009491, -0.0024141, -0.0009207, -0.0014891, 0.0014650
6: -0.0060351, -0.0053509, -0.0060288, -0.0053243, -0.0007107, 0.0006779
7: -0.0031902, -0.0019424, -0.0032847, -0.0019713, -0.0012188, 0.0013423
8: -0.0043876, -0.0015398, -0.0043807, -0.0014545, -0.0029331, 0.0028408
9: 1.0004402, 1.0007747, 1.0004270, 1.0008730, -0.0004327, 0.0003477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003551, upper bound: 0.0003427
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002745, upper bound: 0.0002822
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0040397, -0.0017173, -0.0022255, 0.0023885
1: 0.0049918, 0.0065367, 0.0049900, 0.0066135, -0.0016217, 0.0015466
2: 0.0107026, 0.0150499, 0.0105249, 0.0149022, -0.0036678, 0.0039262
3: -0.0047398, -0.0028598, -0.0047243, -0.0027712, -0.0019686, 0.0018645
4: 0.0045448, 0.0051763, 0.0045530, 0.0052054, -0.0004990, 0.0005911
5: -0.0024483, -0.0009590, -0.0023756, -0.0009070, -0.0015413, 0.0014166
6: -0.0060416, -0.0053530, -0.0060265, -0.0053224, -0.0007192, 0.0006736
7: -0.0032045, -0.0019668, -0.0032559, -0.0019368, -0.0012677, 0.0012891
8: -0.0044411, -0.0015538, -0.0043370, -0.0014382, -0.0030030, 0.0027832
9: 1.0004396, 1.0007656, 1.0004292, 1.0008836, -0.0004439, 0.0003364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003536, upper bound: 0.0003384
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002984, upper bound: 0.0002960
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0040397, -0.0017233, -0.0022197, 0.0023908
1: 0.0049959, 0.0065403, 0.0049960, 0.0066135, -0.0016176, 0.0015443
2: 0.0107029, 0.0150580, 0.0105251, 0.0148918, -0.0036481, 0.0039331
3: -0.0047360, -0.0028568, -0.0047176, -0.0027712, -0.0019648, 0.0018608
4: 0.0045458, 0.0051772, 0.0045551, 0.0052054, -0.0004991, 0.0005850
5: -0.0024551, -0.0009597, -0.0023729, -0.0009075, -0.0015476, 0.0014132
6: -0.0060415, -0.0053525, -0.0060245, -0.0053224, -0.0007191, 0.0006720
7: -0.0032140, -0.0019724, -0.0032559, -0.0019429, -0.0012711, 0.0012836
8: -0.0044474, -0.0015542, -0.0043308, -0.0014384, -0.0030090, 0.0027766
9: 1.0004387, 1.0007666, 1.0004292, 1.0008835, -0.0004448, 0.0003375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003330, upper bound: 0.0003151
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002873, upper bound: 0.0002817
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0040294, -0.0016894, -0.0022534, 0.0023782
1: 0.0049918, 0.0065367, 0.0050022, 0.0066194, -0.0016276, 0.0015345
2: 0.0107026, 0.0150499, 0.0105478, 0.0149702, -0.0037206, 0.0038808
3: -0.0047398, -0.0028598, -0.0047174, -0.0027692, -0.0019706, 0.0018576
4: 0.0045448, 0.0051763, 0.0045534, 0.0052053, -0.0004987, 0.0005972
5: -0.0024483, -0.0009590, -0.0024164, -0.0009202, -0.0015281, 0.0014574
6: -0.0060416, -0.0053530, -0.0060307, -0.0053243, -0.0007173, 0.0006777
7: -0.0032045, -0.0019668, -0.0032847, -0.0019655, -0.0012390, 0.0013178
8: -0.0044411, -0.0015538, -0.0043862, -0.0014543, -0.0029869, 0.0028324
9: 1.0004396, 1.0007656, 1.0004270, 1.0008731, -0.0004334, 0.0003386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003525, upper bound: 0.0003377
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002731, upper bound: 0.0002790
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0040294, -0.0016947, -0.0022484, 0.0023804
1: 0.0049959, 0.0065403, 0.0050078, 0.0066194, -0.0016235, 0.0015326
2: 0.0107029, 0.0150580, 0.0105480, 0.0149617, -0.0037015, 0.0038873
3: -0.0047360, -0.0028568, -0.0047112, -0.0027692, -0.0019668, 0.0018543
4: 0.0045458, 0.0051772, 0.0045554, 0.0052053, -0.0004989, 0.0005910
5: -0.0024551, -0.0009597, -0.0024141, -0.0009207, -0.0015345, 0.0014544
6: -0.0060415, -0.0053525, -0.0060288, -0.0053243, -0.0007171, 0.0006763
7: -0.0032140, -0.0019724, -0.0032847, -0.0019713, -0.0012427, 0.0013123
8: -0.0044474, -0.0015542, -0.0043807, -0.0014545, -0.0029929, 0.0028265
9: 1.0004387, 1.0007666, 1.0004270, 1.0008730, -0.0004343, 0.0003396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003312, upper bound: 0.0003138
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002669, upper bound: 0.0002685
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0041449, -0.0017649, -0.0021862, 0.0024587
1: 0.0049805, 0.0065324, 0.0050150, 0.0066928, -0.0017124, 0.0015174
2: 0.0106823, 0.0149661, 0.0103310, 0.0148092, -0.0036071, 0.0040494
3: -0.0047422, -0.0028611, -0.0046896, -0.0026759, -0.0020663, 0.0018284
4: 0.0045460, 0.0051765, 0.0045649, 0.0052372, -0.0005247, 0.0005758
5: -0.0024038, -0.0009481, -0.0023423, -0.0008497, -0.0015540, 0.0013942
6: -0.0060353, -0.0053514, -0.0060133, -0.0052887, -0.0007465, 0.0006619
7: -0.0031820, -0.0019364, -0.0033235, -0.0019437, -0.0012383, 0.0013871
8: -0.0043820, -0.0015394, -0.0042760, -0.0013106, -0.0030714, 0.0027366
9: 1.0004410, 1.0007735, 1.0004183, 1.0010175, -0.0005765, 0.0003552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003991, upper bound: 0.0003855
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003421, upper bound: 0.0003424
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0041449, -0.0017705, -0.0021810, 0.0024605
1: 0.0049849, 0.0065360, 0.0050206, 0.0066928, -0.0017080, 0.0015154
2: 0.0106827, 0.0149735, 0.0103312, 0.0148001, -0.0035879, 0.0040550
3: -0.0047382, -0.0028582, -0.0046832, -0.0026759, -0.0020623, 0.0018250
4: 0.0045470, 0.0051773, 0.0045669, 0.0052372, -0.0005248, 0.0005698
5: -0.0024098, -0.0009491, -0.0023396, -0.0008503, -0.0015595, 0.0013905
6: -0.0060351, -0.0053509, -0.0060114, -0.0052887, -0.0007463, 0.0006605
7: -0.0031902, -0.0019424, -0.0033235, -0.0019497, -0.0012404, 0.0013811
8: -0.0043876, -0.0015398, -0.0042701, -0.0013108, -0.0030768, 0.0027303
9: 1.0004402, 1.0007747, 1.0004183, 1.0010173, -0.0005771, 0.0003564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003785, upper bound: 0.0003643
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003304, upper bound: 0.0003270
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0041332, -0.0017381, -0.0022130, 0.0024470
1: 0.0049805, 0.0065324, 0.0050250, 0.0066997, -0.0017192, 0.0015073
2: 0.0106823, 0.0149661, 0.0103584, 0.0148763, -0.0036721, 0.0040157
3: -0.0047422, -0.0028611, -0.0046833, -0.0026736, -0.0020686, 0.0018222
4: 0.0045460, 0.0051765, 0.0045654, 0.0052370, -0.0005244, 0.0005761
5: -0.0024038, -0.0009481, -0.0023831, -0.0008633, -0.0015405, 0.0014350
6: -0.0060353, -0.0053514, -0.0060173, -0.0052911, -0.0007441, 0.0006659
7: -0.0031820, -0.0019364, -0.0033504, -0.0019719, -0.0012100, 0.0014140
8: -0.0043820, -0.0015394, -0.0043242, -0.0013300, -0.0030520, 0.0027847
9: 1.0004410, 1.0007735, 1.0004160, 1.0010043, -0.0005634, 0.0003575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003930, upper bound: 0.0003806
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003165, upper bound: 0.0003201
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0041332, -0.0017431, -0.0022084, 0.0024489
1: 0.0049849, 0.0065360, 0.0050303, 0.0066997, -0.0017148, 0.0015057
2: 0.0106827, 0.0149735, 0.0103586, 0.0148680, -0.0036531, 0.0040214
3: -0.0047382, -0.0028582, -0.0046773, -0.0026736, -0.0020646, 0.0018191
4: 0.0045470, 0.0051773, 0.0045672, 0.0052370, -0.0005244, 0.0005700
5: -0.0024098, -0.0009491, -0.0023807, -0.0008638, -0.0015460, 0.0014316
6: -0.0060351, -0.0053509, -0.0060155, -0.0052911, -0.0007439, 0.0006646
7: -0.0031902, -0.0019424, -0.0033504, -0.0019777, -0.0012125, 0.0014080
8: -0.0043876, -0.0015398, -0.0043187, -0.0013302, -0.0030574, 0.0027788
9: 1.0004402, 1.0007747, 1.0004160, 1.0010041, -0.0005639, 0.0003587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003724, upper bound: 0.0003588
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003044, upper bound: 0.0003058
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0041449, -0.0017649, -0.0021779, 0.0024937
1: 0.0049918, 0.0065367, 0.0050150, 0.0066928, -0.0017010, 0.0015217
2: 0.0107026, 0.0150499, 0.0103310, 0.0148092, -0.0035919, 0.0041354
3: -0.0047398, -0.0028598, -0.0046896, -0.0026759, -0.0020639, 0.0018298
4: 0.0045448, 0.0051763, 0.0045649, 0.0052372, -0.0005258, 0.0005756
5: -0.0024483, -0.0009590, -0.0023423, -0.0008497, -0.0015986, 0.0013832
6: -0.0060416, -0.0053530, -0.0060133, -0.0052887, -0.0007529, 0.0006603
7: -0.0032045, -0.0019668, -0.0033235, -0.0019437, -0.0012608, 0.0013567
8: -0.0044411, -0.0015538, -0.0042760, -0.0013106, -0.0031306, 0.0027222
9: 1.0004396, 1.0007656, 1.0004183, 1.0010175, -0.0005778, 0.0003473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003702, upper bound: 0.0003534
time: 1.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003157, upper bound: 0.0003110
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0041449, -0.0017705, -0.0021726, 0.0024959
1: 0.0049959, 0.0065403, 0.0050206, 0.0066928, -0.0016969, 0.0015198
2: 0.0107029, 0.0150580, 0.0103312, 0.0148001, -0.0035728, 0.0041423
3: -0.0047360, -0.0028568, -0.0046832, -0.0026759, -0.0020601, 0.0018264
4: 0.0045458, 0.0051772, 0.0045669, 0.0052372, -0.0005259, 0.0005696
5: -0.0024551, -0.0009597, -0.0023396, -0.0008503, -0.0016049, 0.0013799
6: -0.0060415, -0.0053525, -0.0060114, -0.0052887, -0.0007527, 0.0006588
7: -0.0032140, -0.0019724, -0.0033235, -0.0019497, -0.0012643, 0.0013512
8: -0.0044474, -0.0015542, -0.0042701, -0.0013108, -0.0031367, 0.0027160
9: 1.0004387, 1.0007666, 1.0004183, 1.0010173, -0.0005786, 0.0003483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003486, upper bound: 0.0003307
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003010, upper bound: 0.0002935
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0041332, -0.0017381, -0.0022047, 0.0024820
1: 0.0049918, 0.0065367, 0.0050250, 0.0066997, -0.0017079, 0.0015116
2: 0.0107026, 0.0150499, 0.0103584, 0.0148763, -0.0036381, 0.0040863
3: -0.0047398, -0.0028598, -0.0046833, -0.0026736, -0.0020662, 0.0018235
4: 0.0045448, 0.0051763, 0.0045654, 0.0052370, -0.0005255, 0.0005803
5: -0.0024483, -0.0009590, -0.0023831, -0.0008633, -0.0015850, 0.0014241
6: -0.0060416, -0.0053530, -0.0060173, -0.0052911, -0.0007505, 0.0006643
7: -0.0032045, -0.0019668, -0.0033504, -0.0019719, -0.0012325, 0.0013836
8: -0.0044411, -0.0015538, -0.0043242, -0.0013300, -0.0031111, 0.0027704
9: 1.0004396, 1.0007656, 1.0004160, 1.0010043, -0.0005647, 0.0003495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003699, upper bound: 0.0003534
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003044, upper bound: 0.0003038
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0041332, -0.0017431, -0.0021999, 0.0024843
1: 0.0049959, 0.0065403, 0.0050303, 0.0066997, -0.0017038, 0.0015100
2: 0.0107029, 0.0150580, 0.0103586, 0.0148680, -0.0036192, 0.0040928
3: -0.0047360, -0.0028568, -0.0046773, -0.0026736, -0.0020624, 0.0018205
4: 0.0045458, 0.0051772, 0.0045672, 0.0052370, -0.0005257, 0.0005741
5: -0.0024551, -0.0009597, -0.0023807, -0.0008638, -0.0015914, 0.0014210
6: -0.0060415, -0.0053525, -0.0060155, -0.0052911, -0.0007504, 0.0006629
7: -0.0032140, -0.0019724, -0.0033504, -0.0019777, -0.0012364, 0.0013780
8: -0.0044474, -0.0015542, -0.0043187, -0.0013302, -0.0031172, 0.0027645
9: 1.0004387, 1.0007666, 1.0004160, 1.0010041, -0.0005654, 0.0003506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003478, upper bound: 0.0003300
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002919, upper bound: 0.0002877
time: 0.75 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.89 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004695, upper bound: 0.0004695
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004668, upper bound: 0.0004680
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004680, upper bound: 0.0004668
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0004660, upper bound: 0.0004660
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003991, upper bound: 0.0003856
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003428, upper bound: 0.0003428
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003785, upper bound: 0.0003644
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003315, upper bound: 0.0003278
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003930, upper bound: 0.0003806
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003182, upper bound: 0.0003224
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003725, upper bound: 0.0003589
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003062, upper bound: 0.0003079
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003702, upper bound: 0.0003535
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003173, upper bound: 0.0003126
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003487, upper bound: 0.0003311
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003036, upper bound: 0.0002952
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003699, upper bound: 0.0003534
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003071, upper bound: 0.0003067
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003478, upper bound: 0.0003305
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002947, upper bound: 0.0002901
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003831, upper bound: 0.0003706
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003232, upper bound: 0.0003273
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003639, upper bound: 0.0003496
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003147, upper bound: 0.0003156
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003734, upper bound: 0.0003629
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002817, upper bound: 0.0002919
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003551, upper bound: 0.0003427
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002745, upper bound: 0.0002822
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003536, upper bound: 0.0003384
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002984, upper bound: 0.0002960
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003330, upper bound: 0.0003151
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002873, upper bound: 0.0002817
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003525, upper bound: 0.0003377
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002731, upper bound: 0.0002790
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003312, upper bound: 0.0003138
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002669, upper bound: 0.0002685
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003991, upper bound: 0.0003855
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003421, upper bound: 0.0003424
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003785, upper bound: 0.0003643
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003304, upper bound: 0.0003270
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003930, upper bound: 0.0003806
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003165, upper bound: 0.0003201
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003724, upper bound: 0.0003588
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003044, upper bound: 0.0003058
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003702, upper bound: 0.0003534
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003157, upper bound: 0.0003110
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003486, upper bound: 0.0003307
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003010, upper bound: 0.0002935
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003699, upper bound: 0.0003534
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003044, upper bound: 0.0003038
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0003478, upper bound: 0.0003300
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.89
Output dim: 9, lower bound: -0.0002919, upper bound: 0.0002877

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0039512, -0.0016862, -0.0022649, 0.0022649
1: 0.0049805, 0.0065324, 0.0049805, 0.0065324, -0.0015519, 0.0015519
2: 0.0106823, 0.0149661, 0.0106823, 0.0149661, -0.0036183, 0.0036183
3: -0.0047422, -0.0028611, -0.0047422, -0.0028611, -0.0018810, 0.0018810
4: 0.0045460, 0.0051765, 0.0045460, 0.0051765, -0.0004218, 0.0004218
5: -0.0024038, -0.0009481, -0.0024038, -0.0009481, -0.0014557, 0.0014557
6: -0.0060353, -0.0053514, -0.0060353, -0.0053514, -0.0006839, 0.0006839
7: -0.0031820, -0.0019364, -0.0031820, -0.0019364, -0.0012455, 0.0012455
8: -0.0043820, -0.0015394, -0.0043820, -0.0015394, -0.0028426, 0.0028426
9: 1.0004410, 1.0007735, 1.0004410, 1.0007735, -0.0003326, 0.0003326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003537, upper bound: 0.0003667
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003261, upper bound: 0.0003297
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0039515, -0.0016843, -0.0022668, 0.0022653
1: 0.0049805, 0.0065324, 0.0049849, 0.0065360, -0.0015555, 0.0015475
2: 0.0106823, 0.0149661, 0.0106827, 0.0149735, -0.0036247, 0.0036177
3: -0.0047422, -0.0028611, -0.0047382, -0.0028582, -0.0018840, 0.0018771
4: 0.0045460, 0.0051765, 0.0045470, 0.0051773, -0.0004231, 0.0004213
5: -0.0024038, -0.0009481, -0.0024098, -0.0009491, -0.0014547, 0.0014617
6: -0.0060353, -0.0053514, -0.0060351, -0.0053509, -0.0006844, 0.0006837
7: -0.0031820, -0.0019364, -0.0031902, -0.0019424, -0.0012396, 0.0012537
8: -0.0043820, -0.0015394, -0.0043876, -0.0015398, -0.0028421, 0.0028482
9: 1.0004410, 1.0007735, 1.0004402, 1.0007747, -0.0003338, 0.0003333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003537, upper bound: 0.0003667
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003261, upper bound: 0.0003297
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0039512, -0.0016862, -0.0022653, 0.0022668
1: 0.0049849, 0.0065360, 0.0049805, 0.0065324, -0.0015475, 0.0015555
2: 0.0106827, 0.0149735, 0.0106823, 0.0149661, -0.0036177, 0.0036247
3: -0.0047382, -0.0028582, -0.0047422, -0.0028611, -0.0018771, 0.0018840
4: 0.0045470, 0.0051773, 0.0045460, 0.0051765, -0.0004213, 0.0004231
5: -0.0024098, -0.0009491, -0.0024038, -0.0009481, -0.0014617, 0.0014547
6: -0.0060351, -0.0053509, -0.0060353, -0.0053514, -0.0006837, 0.0006844
7: -0.0031902, -0.0019424, -0.0031820, -0.0019364, -0.0012537, 0.0012396
8: -0.0043876, -0.0015398, -0.0043820, -0.0015394, -0.0028482, 0.0028421
9: 1.0004402, 1.0007747, 1.0004410, 1.0007735, -0.0003333, 0.0003338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003504, upper bound: 0.0003644
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003174, upper bound: 0.0003174
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0039515, -0.0016843, -0.0022671, 0.0022671
1: 0.0049849, 0.0065360, 0.0049849, 0.0065360, -0.0015511, 0.0015511
2: 0.0106827, 0.0149735, 0.0106827, 0.0149735, -0.0036236, 0.0036236
3: -0.0047382, -0.0028582, -0.0047382, -0.0028582, -0.0018800, 0.0018800
4: 0.0045470, 0.0051773, 0.0045470, 0.0051773, -0.0004219, 0.0004219
5: -0.0024098, -0.0009491, -0.0024098, -0.0009491, -0.0014607, 0.0014607
6: -0.0060351, -0.0053509, -0.0060351, -0.0053509, -0.0006841, 0.0006841
7: -0.0031902, -0.0019424, -0.0031902, -0.0019424, -0.0012478, 0.0012478
8: -0.0043876, -0.0015398, -0.0043876, -0.0015398, -0.0028478, 0.0028478
9: 1.0004402, 1.0007747, 1.0004402, 1.0007747, -0.0003345, 0.0003345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003504, upper bound: 0.0003644
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003174, upper bound: 0.0003174
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0039428, -0.0016512, -0.0023000, 0.0022566
1: 0.0049805, 0.0065324, 0.0049918, 0.0065367, -0.0015562, 0.0015406
2: 0.0106823, 0.0149661, 0.0107026, 0.0150499, -0.0037043, 0.0036027
3: -0.0047422, -0.0028611, -0.0047398, -0.0028598, -0.0018824, 0.0018787
4: 0.0045460, 0.0051765, 0.0045448, 0.0051763, -0.0004215, 0.0004229
5: -0.0024038, -0.0009481, -0.0024483, -0.0009590, -0.0014447, 0.0015002
6: -0.0060353, -0.0053514, -0.0060416, -0.0053530, -0.0006823, 0.0006903
7: -0.0031820, -0.0019364, -0.0032045, -0.0019668, -0.0012151, 0.0012680
8: -0.0043820, -0.0015394, -0.0044411, -0.0015538, -0.0028282, 0.0029017
9: 1.0004410, 1.0007735, 1.0004396, 1.0007656, -0.0003246, 0.0003339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003196, upper bound: 0.0003366
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002931, upper bound: 0.0003022
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039512, -0.0016862, -0.0039431, -0.0016490, -0.0023022, 0.0022568
1: 0.0049805, 0.0065324, 0.0049959, 0.0065403, -0.0015599, 0.0015364
2: 0.0106823, 0.0149661, 0.0107029, 0.0150580, -0.0037120, 0.0036022
3: -0.0047422, -0.0028611, -0.0047360, -0.0028568, -0.0018854, 0.0018748
4: 0.0045460, 0.0051765, 0.0045458, 0.0051772, -0.0004229, 0.0004224
5: -0.0024038, -0.0009481, -0.0024551, -0.0009597, -0.0014441, 0.0015070
6: -0.0060353, -0.0053514, -0.0060415, -0.0053525, -0.0006827, 0.0006901
7: -0.0031820, -0.0019364, -0.0032140, -0.0019724, -0.0012096, 0.0012776
8: -0.0043820, -0.0015394, -0.0044474, -0.0015542, -0.0028278, 0.0029080
9: 1.0004410, 1.0007735, 1.0004387, 1.0007666, -0.0003257, 0.0003349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003196, upper bound: 0.0003366
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002931, upper bound: 0.0003022
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0039428, -0.0016512, -0.0023003, 0.0022584
1: 0.0049849, 0.0065360, 0.0049918, 0.0065367, -0.0015518, 0.0015442
2: 0.0106827, 0.0149735, 0.0107026, 0.0150499, -0.0037037, 0.0036090
3: -0.0047382, -0.0028582, -0.0047398, -0.0028598, -0.0018784, 0.0018816
4: 0.0045470, 0.0051773, 0.0045448, 0.0051763, -0.0004211, 0.0004242
5: -0.0024098, -0.0009491, -0.0024483, -0.0009590, -0.0014507, 0.0014993
6: -0.0060351, -0.0053509, -0.0060416, -0.0053530, -0.0006821, 0.0006907
7: -0.0031902, -0.0019424, -0.0032045, -0.0019668, -0.0012233, 0.0012621
8: -0.0043876, -0.0015398, -0.0044411, -0.0015538, -0.0028338, 0.0029013
9: 1.0004402, 1.0007747, 1.0004396, 1.0007656, -0.0003253, 0.0003351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003164, upper bound: 0.0003341
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002855, upper bound: 0.0002921
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039515, -0.0016843, -0.0039431, -0.0016490, -0.0023025, 0.0022587
1: 0.0049849, 0.0065360, 0.0049959, 0.0065403, -0.0015555, 0.0015401
2: 0.0106827, 0.0149735, 0.0107029, 0.0150580, -0.0037105, 0.0036078
3: -0.0047382, -0.0028582, -0.0047360, -0.0028568, -0.0018814, 0.0018778
4: 0.0045470, 0.0051773, 0.0045458, 0.0051772, -0.0004216, 0.0004230
5: -0.0024098, -0.0009491, -0.0024551, -0.0009597, -0.0014501, 0.0015061
6: -0.0060351, -0.0053509, -0.0060415, -0.0053525, -0.0006825, 0.0006906
7: -0.0031902, -0.0019424, -0.0032140, -0.0019724, -0.0012178, 0.0012716
8: -0.0043876, -0.0015398, -0.0044474, -0.0015542, -0.0028334, 0.0029076
9: 1.0004402, 1.0007747, 1.0004387, 1.0007666, -0.0003264, 0.0003361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003164, upper bound: 0.0003341
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0002855, upper bound: 0.0002921
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0039512, -0.0016862, -0.0022566, 0.0023000
1: 0.0049918, 0.0065367, 0.0049805, 0.0065324, -0.0015406, 0.0015562
2: 0.0107026, 0.0150499, 0.0106823, 0.0149661, -0.0036027, 0.0037043
3: -0.0047398, -0.0028598, -0.0047422, -0.0028611, -0.0018787, 0.0018824
4: 0.0045448, 0.0051763, 0.0045460, 0.0051765, -0.0004229, 0.0004215
5: -0.0024483, -0.0009590, -0.0024038, -0.0009481, -0.0015002, 0.0014447
6: -0.0060416, -0.0053530, -0.0060353, -0.0053514, -0.0006903, 0.0006823
7: -0.0032045, -0.0019668, -0.0031820, -0.0019364, -0.0012680, 0.0012151
8: -0.0044411, -0.0015538, -0.0043820, -0.0015394, -0.0029017, 0.0028282
9: 1.0004396, 1.0007656, 1.0004410, 1.0007735, -0.0003339, 0.0003246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004121, upper bound: 0.0004203
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003971, upper bound: 0.0003977
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0039515, -0.0016843, -0.0022584, 0.0023003
1: 0.0049918, 0.0065367, 0.0049849, 0.0065360, -0.0015442, 0.0015518
2: 0.0107026, 0.0150499, 0.0106827, 0.0149735, -0.0036090, 0.0037037
3: -0.0047398, -0.0028598, -0.0047382, -0.0028582, -0.0018816, 0.0018784
4: 0.0045448, 0.0051763, 0.0045470, 0.0051773, -0.0004242, 0.0004211
5: -0.0024483, -0.0009590, -0.0024098, -0.0009491, -0.0014993, 0.0014507
6: -0.0060416, -0.0053530, -0.0060351, -0.0053509, -0.0006907, 0.0006821
7: -0.0032045, -0.0019668, -0.0031902, -0.0019424, -0.0012621, 0.0012233
8: -0.0044411, -0.0015538, -0.0043876, -0.0015398, -0.0029013, 0.0028338
9: 1.0004396, 1.0007656, 1.0004402, 1.0007747, -0.0003351, 0.0003253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004121, upper bound: 0.0004203
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003971, upper bound: 0.0003977
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0039512, -0.0016862, -0.0022568, 0.0023022
1: 0.0049959, 0.0065403, 0.0049805, 0.0065324, -0.0015364, 0.0015599
2: 0.0107029, 0.0150580, 0.0106823, 0.0149661, -0.0036022, 0.0037120
3: -0.0047360, -0.0028568, -0.0047422, -0.0028611, -0.0018748, 0.0018854
4: 0.0045458, 0.0051772, 0.0045460, 0.0051765, -0.0004224, 0.0004229
5: -0.0024551, -0.0009597, -0.0024038, -0.0009481, -0.0015070, 0.0014441
6: -0.0060415, -0.0053525, -0.0060353, -0.0053514, -0.0006901, 0.0006827
7: -0.0032140, -0.0019724, -0.0031820, -0.0019364, -0.0012776, 0.0012096
8: -0.0044474, -0.0015542, -0.0043820, -0.0015394, -0.0029080, 0.0028278
9: 1.0004387, 1.0007666, 1.0004410, 1.0007735, -0.0003349, 0.0003257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004110, upper bound: 0.0004191
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003906, upper bound: 0.0003899
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0039515, -0.0016843, -0.0022587, 0.0023025
1: 0.0049959, 0.0065403, 0.0049849, 0.0065360, -0.0015401, 0.0015555
2: 0.0107029, 0.0150580, 0.0106827, 0.0149735, -0.0036078, 0.0037105
3: -0.0047360, -0.0028568, -0.0047382, -0.0028582, -0.0018778, 0.0018814
4: 0.0045458, 0.0051772, 0.0045470, 0.0051773, -0.0004230, 0.0004216
5: -0.0024551, -0.0009597, -0.0024098, -0.0009491, -0.0015061, 0.0014501
6: -0.0060415, -0.0053525, -0.0060351, -0.0053509, -0.0006906, 0.0006825
7: -0.0032140, -0.0019724, -0.0031902, -0.0019424, -0.0012716, 0.0012178
8: -0.0044474, -0.0015542, -0.0043876, -0.0015398, -0.0029076, 0.0028334
9: 1.0004387, 1.0007666, 1.0004402, 1.0007747, -0.0003361, 0.0003264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004110, upper bound: 0.0004191
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003906, upper bound: 0.0003899
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0039428, -0.0016512, -0.0022916, 0.0022916
1: 0.0049918, 0.0065367, 0.0049918, 0.0065367, -0.0015449, 0.0015449
2: 0.0107026, 0.0150499, 0.0107026, 0.0150499, -0.0036616, 0.0036616
3: -0.0047398, -0.0028598, -0.0047398, -0.0028598, -0.0018800, 0.0018800
4: 0.0045448, 0.0051763, 0.0045448, 0.0051763, -0.0004226, 0.0004226
5: -0.0024483, -0.0009590, -0.0024483, -0.0009590, -0.0014893, 0.0014893
6: -0.0060416, -0.0053530, -0.0060416, -0.0053530, -0.0006887, 0.0006887
7: -0.0032045, -0.0019668, -0.0032045, -0.0019668, -0.0012377, 0.0012377
8: -0.0044411, -0.0015538, -0.0044411, -0.0015538, -0.0028873, 0.0028873
9: 1.0004396, 1.0007656, 1.0004396, 1.0007656, -0.0003259, 0.0003259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004052, upper bound: 0.0004143
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003931, upper bound: 0.0003950
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039428, -0.0016512, -0.0039431, -0.0016490, -0.0022938, 0.0022919
1: 0.0049918, 0.0065367, 0.0049959, 0.0065403, -0.0015485, 0.0015407
2: 0.0107026, 0.0150499, 0.0107029, 0.0150580, -0.0036688, 0.0036610
3: -0.0047398, -0.0028598, -0.0047360, -0.0028568, -0.0018830, 0.0018762
4: 0.0045448, 0.0051763, 0.0045458, 0.0051772, -0.0004240, 0.0004222
5: -0.0024483, -0.0009590, -0.0024551, -0.0009597, -0.0014886, 0.0014961
6: -0.0060416, -0.0053530, -0.0060415, -0.0053525, -0.0006891, 0.0006885
7: -0.0032045, -0.0019668, -0.0032140, -0.0019724, -0.0012321, 0.0012472
8: -0.0044411, -0.0015538, -0.0044474, -0.0015542, -0.0028869, 0.0028936
9: 1.0004396, 1.0007656, 1.0004387, 1.0007666, -0.0003270, 0.0003269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004052, upper bound: 0.0004143
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003931, upper bound: 0.0003950
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0039428, -0.0016512, -0.0022919, 0.0022938
1: 0.0049959, 0.0065403, 0.0049918, 0.0065367, -0.0015407, 0.0015485
2: 0.0107029, 0.0150580, 0.0107026, 0.0150499, -0.0036610, 0.0036688
3: -0.0047360, -0.0028568, -0.0047398, -0.0028598, -0.0018762, 0.0018830
4: 0.0045458, 0.0051772, 0.0045448, 0.0051763, -0.0004222, 0.0004240
5: -0.0024551, -0.0009597, -0.0024483, -0.0009590, -0.0014961, 0.0014886
6: -0.0060415, -0.0053525, -0.0060416, -0.0053530, -0.0006885, 0.0006891
7: -0.0032140, -0.0019724, -0.0032045, -0.0019668, -0.0012472, 0.0012321
8: -0.0044474, -0.0015542, -0.0044411, -0.0015538, -0.0028936, 0.0028869
9: 1.0004387, 1.0007666, 1.0004396, 1.0007656, -0.0003269, 0.0003270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004038, upper bound: 0.0004130
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003864, upper bound: 0.0003864
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039431, -0.0016490, -0.0039431, -0.0016490, -0.0022941, 0.0022941
1: 0.0049959, 0.0065403, 0.0049959, 0.0065403, -0.0015444, 0.0015444
2: 0.0107029, 0.0150580, 0.0107029, 0.0150580, -0.0036679, 0.0036679
3: -0.0047360, -0.0028568, -0.0047360, -0.0028568, -0.0018792, 0.0018792
4: 0.0045458, 0.0051772, 0.0045458, 0.0051772, -0.0004228, 0.0004228
5: -0.0024551, -0.0009597, -0.0024551, -0.0009597, -0.0014954, 0.0014954
6: -0.0060415, -0.0053525, -0.0060415, -0.0053525, -0.0006889, 0.0006889
7: -0.0032140, -0.0019724, -0.0032140, -0.0019724, -0.0012417, 0.0012417
8: -0.0044474, -0.0015542, -0.0044474, -0.0015542, -0.0028933, 0.0028933
9: 1.0004387, 1.0007666, 1.0004387, 1.0007666, -0.0003279, 0.0003279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 212

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0004038, upper bound: 0.0004130
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0003864, upper bound: 0.0003864
time: 0.69 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.72 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003537, upper bound: 0.0003667
IS_A1_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003261, upper bound: 0.0003297
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003537, upper bound: 0.0003667
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003261, upper bound: 0.0003297
IS_A1_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003504, upper bound: 0.0003644
IS_A1_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003174, upper bound: 0.0003174
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003504, upper bound: 0.0003644
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003174, upper bound: 0.0003174
IS_A1_B1_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003196, upper bound: 0.0003366
IS_A1_B1_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0002931, upper bound: 0.0003022
IS_A1_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003196, upper bound: 0.0003366
IS_A1_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0002931, upper bound: 0.0003022
IS_A1_B1_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003164, upper bound: 0.0003341
IS_A1_B1_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0002855, upper bound: 0.0002921
IS_A1_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003164, upper bound: 0.0003341
IS_A1_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0002855, upper bound: 0.0002921
IS_A1_B1_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004121, upper bound: 0.0004203
IS_A1_B1_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003971, upper bound: 0.0003977
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004121, upper bound: 0.0004203
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003971, upper bound: 0.0003977
IS_A1_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004110, upper bound: 0.0004191
IS_A1_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003906, upper bound: 0.0003899
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004110, upper bound: 0.0004191
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003906, upper bound: 0.0003899
IS_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004052, upper bound: 0.0004143
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003931, upper bound: 0.0003950
IS_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004052, upper bound: 0.0004143
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003931, upper bound: 0.0003950
IS_A1_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004038, upper bound: 0.0004130
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003864, upper bound: 0.0003864
IS_A1_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0004038, upper bound: 0.0004130
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.72
Output dim: 9, lower bound: -0.0003864, upper bound: 0.0003864

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.93 + 495.75 = 498.69 seconds
