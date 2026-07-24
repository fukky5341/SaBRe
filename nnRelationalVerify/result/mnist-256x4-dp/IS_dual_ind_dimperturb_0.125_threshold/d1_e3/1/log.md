## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00010188


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040966, -0.0040883, -0.0040966, -0.0040883, -0.0000036, 0.0000036)
1: (-0.0060751, -0.0057616, -0.0060751, -0.0057616, -0.0001334, 0.0001334)
2: (0.9691731, 0.9695493, 0.9691731, 0.9695493, -0.0001601, 0.0001601)
3: (0.0189314, 0.0217067, 0.0189314, 0.0217067, -0.0011806, 0.0011806)
4: (-0.0023439, -0.0021329, -0.0023439, -0.0021329, -0.0000898, 0.0000898)
5: (0.0149014, 0.0151147, 0.0149014, 0.0151147, -0.0000908, 0.0000908)
6: (0.0045550, 0.0046588, 0.0045550, 0.0046588, -0.0000441, 0.0000441)
7: (-0.0134037, -0.0126845, -0.0134037, -0.0126845, -0.0003060, 0.0003060)
8: (0.0060953, 0.0066659, 0.0060953, 0.0066659, -0.0002427, 0.0002427)
9: (0.0086875, 0.0097138, 0.0086875, 0.0097138, -0.0004366, 0.0004366)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.24 = 2.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0001095, upper bound: 0.0001095

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0000992, upper bound: 0.0001061
time: 0.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001064, upper bound: 0.0001064
time: 0.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.01 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 2, lower bound: -0.0000992, upper bound: 0.0001061
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 2, lower bound: -0.0001064, upper bound: 0.0001064

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0040963, -0.0040883, -0.0040965, -0.0040883, -0.0000033, 0.0000034
1: -0.0060627, -0.0057639, -0.0060709, -0.0057616, -0.0001217, 0.0001274
2: 0.9691880, 0.9695466, 0.9691781, 0.9695492, -0.0001461, 0.0001529
3: 0.0190412, 0.0216864, 0.0189694, 0.0217063, -0.0010773, 0.0011280
4: -0.0023424, -0.0021412, -0.0023439, -0.0021358, -0.0000858, 0.0000819
5: 0.0149029, 0.0151063, 0.0149014, 0.0151118, -0.0000867, 0.0000828
6: 0.0045591, 0.0046580, 0.0045564, 0.0046587, -0.0000403, 0.0000422
7: -0.0133985, -0.0127129, -0.0134036, -0.0126943, -0.0002923, 0.0002792
8: 0.0060994, 0.0066433, 0.0060953, 0.0066581, -0.0002319, 0.0002215
9: 0.0086950, 0.0096732, 0.0086877, 0.0096998, -0.0004171, 0.0003984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000992, upper bound: 0.0000992
time: 0.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0000992, upper bound: 0.0001061
time: 0.42 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0040883, -0.0040966, -0.0040883, -0.0000033, 0.0000035
1: -0.0060711, -0.0057617, -0.0060742, -0.0057616, -0.0001230, 0.0001325
2: 0.9691779, 0.9695492, 0.9691742, 0.9695492, -0.0001475, 0.0001590
3: 0.0189671, 0.0217062, 0.0189402, 0.0217065, -0.0010883, 0.0011727
4: -0.0023439, -0.0021356, -0.0023439, -0.0021335, -0.0000892, 0.0000828
5: 0.0149014, 0.0151120, 0.0149014, 0.0151140, -0.0000901, 0.0000837
6: 0.0045563, 0.0046587, 0.0045553, 0.0046588, -0.0000407, 0.0000438
7: -0.0134036, -0.0126937, -0.0134037, -0.0126868, -0.0003039, 0.0002820
8: 0.0060954, 0.0066585, 0.0060953, 0.0066641, -0.0002411, 0.0002238
9: 0.0086877, 0.0097006, 0.0086876, 0.0097106, -0.0004337, 0.0004024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001061, upper bound: 0.0000992
time: 0.42 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001061, upper bound: 0.0001064
time: 0.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.26 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.26
Output dim: 2, lower bound: -0.0000992, upper bound: 0.0000992
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 2, lower bound: -0.0000992, upper bound: 0.0001061
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 2, lower bound: -0.0001061, upper bound: 0.0000992
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 2, lower bound: -0.0001061, upper bound: 0.0001064

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040963, -0.0040883, -0.0040965, -0.0040883, -0.0000032, 0.0000034
1: -0.0060627, -0.0057639, -0.0060711, -0.0057617, -0.0001217, 0.0001290
2: 0.9691880, 0.9695466, 0.9691779, 0.9695492, -0.0001461, 0.0001548
3: 0.0190412, 0.0216864, 0.0189671, 0.0217062, -0.0010773, 0.0011417
4: -0.0023424, -0.0021412, -0.0023439, -0.0021356, -0.0000868, 0.0000819
5: 0.0149029, 0.0151063, 0.0149014, 0.0151120, -0.0000878, 0.0000828
6: 0.0045591, 0.0046580, 0.0045563, 0.0046587, -0.0000403, 0.0000427
7: -0.0133985, -0.0127129, -0.0134036, -0.0126937, -0.0002959, 0.0002792
8: 0.0060994, 0.0066433, 0.0060954, 0.0066585, -0.0002347, 0.0002215
9: 0.0086950, 0.0096732, 0.0086877, 0.0097006, -0.0004222, 0.0003984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000929, upper bound: 0.0001010
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0000951, upper bound: 0.0001022
time: 0.44 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0040883, -0.0040963, -0.0040883, -0.0000034, 0.0000032
1: -0.0060711, -0.0057617, -0.0060627, -0.0057639, -0.0001290, 0.0001217
2: 0.9691779, 0.9695492, 0.9691880, 0.9695466, -0.0001548, 0.0001461
3: 0.0189671, 0.0217062, 0.0190412, 0.0216864, -0.0011417, 0.0010773
4: -0.0023439, -0.0021356, -0.0023424, -0.0021412, -0.0000819, 0.0000868
5: 0.0149014, 0.0151120, 0.0149029, 0.0151063, -0.0000828, 0.0000878
6: 0.0045563, 0.0046587, 0.0045591, 0.0046580, -0.0000427, 0.0000403
7: -0.0134036, -0.0126937, -0.0133985, -0.0127129, -0.0002792, 0.0002959
8: 0.0060954, 0.0066585, 0.0060994, 0.0066433, -0.0002215, 0.0002347
9: 0.0086877, 0.0097006, 0.0086950, 0.0096732, -0.0003984, 0.0004222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000991, upper bound: 0.0000936
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001022, upper bound: 0.0000951
time: 0.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0040883, -0.0040965, -0.0040883, -0.0000033, 0.0000033
1: -0.0060711, -0.0057617, -0.0060711, -0.0057617, -0.0001229, 0.0001229
2: 0.9691779, 0.9695492, 0.9691779, 0.9695492, -0.0001475, 0.0001475
3: 0.0189671, 0.0217062, 0.0189671, 0.0217062, -0.0010881, 0.0010881
4: -0.0023439, -0.0021356, -0.0023439, -0.0021356, -0.0000828, 0.0000828
5: 0.0149014, 0.0151120, 0.0149014, 0.0151120, -0.0000836, 0.0000836
6: 0.0045563, 0.0046587, 0.0045563, 0.0046587, -0.0000407, 0.0000407
7: -0.0134036, -0.0126937, -0.0134036, -0.0126937, -0.0002820, 0.0002820
8: 0.0060954, 0.0066585, 0.0060954, 0.0066585, -0.0002237, 0.0002237
9: 0.0086877, 0.0097006, 0.0086877, 0.0097006, -0.0004024, 0.0004024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000991, upper bound: 0.0000946
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001022, upper bound: 0.0000955
time: 0.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.17 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 2, lower bound: -0.0000929, upper bound: 0.0001010
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 2, lower bound: -0.0000951, upper bound: 0.0001022
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 2, lower bound: -0.0000991, upper bound: 0.0000936
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 2, lower bound: -0.0001022, upper bound: 0.0000951
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 2, lower bound: -0.0000991, upper bound: 0.0000946
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 2, lower bound: -0.0001022, upper bound: 0.0000955

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040963, -0.0040883, -0.0040965, -0.0040883, -0.0000028, 0.0000034
1: -0.0060606, -0.0057639, -0.0060711, -0.0057617, -0.0001031, 0.0001286
2: 0.9691905, 0.9695466, 0.9691779, 0.9695492, -0.0001237, 0.0001543
3: 0.0190605, 0.0216864, 0.0189671, 0.0217062, -0.0009122, 0.0011384
4: -0.0023424, -0.0021427, -0.0023439, -0.0021356, -0.0000866, 0.0000694
5: 0.0149029, 0.0151048, 0.0149014, 0.0151120, -0.0000875, 0.0000701
6: 0.0045598, 0.0046580, 0.0045563, 0.0046587, -0.0000341, 0.0000426
7: -0.0133985, -0.0127179, -0.0134036, -0.0126937, -0.0002950, 0.0002364
8: 0.0060994, 0.0066393, 0.0060954, 0.0066585, -0.0002341, 0.0001876
9: 0.0086950, 0.0096661, 0.0086877, 0.0097006, -0.0004210, 0.0003373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000936, upper bound: 0.0000992
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0000936, upper bound: 0.0001022
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0040883, -0.0040963, -0.0040883, -0.0000029, 0.0000032
1: -0.0060689, -0.0057617, -0.0060627, -0.0057639, -0.0001093, 0.0001214
2: 0.9691805, 0.9695492, 0.9691880, 0.9695466, -0.0001312, 0.0001456
3: 0.0189863, 0.0217062, 0.0190412, 0.0216864, -0.0009674, 0.0010742
4: -0.0023439, -0.0021370, -0.0023424, -0.0021412, -0.0000817, 0.0000736
5: 0.0149014, 0.0151105, 0.0149029, 0.0151063, -0.0000826, 0.0000744
6: 0.0045571, 0.0046587, 0.0045591, 0.0046580, -0.0000362, 0.0000402
7: -0.0134036, -0.0126987, -0.0133985, -0.0127129, -0.0002784, 0.0002507
8: 0.0060954, 0.0066546, 0.0060994, 0.0066433, -0.0002209, 0.0001989
9: 0.0086877, 0.0096935, 0.0086950, 0.0096732, -0.0003972, 0.0003578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001010, upper bound: 0.0000929
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001010, upper bound: 0.0000951
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040965, -0.0040883, -0.0040965, -0.0040883, -0.0000027, 0.0000033
1: -0.0060689, -0.0057617, -0.0060711, -0.0057617, -0.0001012, 0.0001226
2: 0.9691805, 0.9695492, 0.9691779, 0.9695492, -0.0001214, 0.0001471
3: 0.0189863, 0.0217062, 0.0189671, 0.0217062, -0.0008954, 0.0010848
4: -0.0023439, -0.0021370, -0.0023439, -0.0021356, -0.0000825, 0.0000681
5: 0.0149014, 0.0151105, 0.0149014, 0.0151120, -0.0000834, 0.0000688
6: 0.0045571, 0.0046587, 0.0045563, 0.0046587, -0.0000335, 0.0000406
7: -0.0134036, -0.0126987, -0.0134036, -0.0126937, -0.0002811, 0.0002320
8: 0.0060954, 0.0066546, 0.0060954, 0.0066585, -0.0002230, 0.0001841
9: 0.0086877, 0.0096935, 0.0086877, 0.0097006, -0.0004012, 0.0003311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001016, upper bound: 0.0000939
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001016, upper bound: 0.0000955
time: 0.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.59 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 2, lower bound: -0.0000936, upper bound: 0.0000992
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 2, lower bound: -0.0000936, upper bound: 0.0001022
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 2, lower bound: -0.0001010, upper bound: 0.0000929
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 2, lower bound: -0.0001010, upper bound: 0.0000951
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 2, lower bound: -0.0001016, upper bound: 0.0000939
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 2, lower bound: -0.0001016, upper bound: 0.0000955

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040963, -0.0040883, -0.0040965, -0.0040883, -0.0000027, 0.0000029
1: -0.0060606, -0.0057639, -0.0060689, -0.0057617, -0.0001029, 0.0001091
2: 0.9691905, 0.9695466, 0.9691805, 0.9695492, -0.0001235, 0.0001309
3: 0.0190605, 0.0216864, 0.0189863, 0.0217062, -0.0009110, 0.0009656
4: -0.0023424, -0.0021427, -0.0023439, -0.0021370, -0.0000734, 0.0000693
5: 0.0149029, 0.0151048, 0.0149014, 0.0151105, -0.0000742, 0.0000700
6: 0.0045598, 0.0046580, 0.0045571, 0.0046587, -0.0000341, 0.0000361
7: -0.0133985, -0.0127179, -0.0134036, -0.0126987, -0.0002502, 0.0002361
8: 0.0060994, 0.0066393, 0.0060954, 0.0066546, -0.0001985, 0.0001873
9: 0.0086950, 0.0096661, 0.0086877, 0.0096935, -0.0003571, 0.0003369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 65

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000577, upper bound: 0.0000974
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0000601, upper bound: 0.0000974
time: 0.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.66 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0000577, upper bound: 0.0000974
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0000601, upper bound: 0.0000974

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.57 + 21.85 = 24.42 seconds
