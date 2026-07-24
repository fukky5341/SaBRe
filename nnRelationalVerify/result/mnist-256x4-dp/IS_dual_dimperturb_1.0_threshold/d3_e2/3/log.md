## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 9.007671511800002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640)
1: (-6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886)
2: (-7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480)
3: (-9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012)
4: (-8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819)
5: (-6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571)
6: (-6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572)
7: (-8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652)
8: (-8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467)
9: (-6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 4.96 = 6.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -9.0166882, upper bound: 9.0166883

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0139124, upper bound: 9.0099656
time: 6.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
time: 3.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.94
Output dim: 7, lower bound: -9.0139124, upper bound: 9.0099656
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.94
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.5955572, 5.8445797, -7.6137094, 5.8590560, -13.4546127, 13.4582891
1: -6.1136093, 5.2946382, -6.1292791, 5.3075094, -11.4211159, 11.4239178
2: -7.4492431, 4.5000267, -7.4691033, 4.5115442, -11.9607868, 11.9691296
3: -9.0429592, 4.4055662, -9.0654783, 4.4164257, -13.4593840, 13.4710407
4: -8.1823597, 6.3783607, -8.2032833, 6.3942995, -14.5766592, 14.5816441
5: -6.5400386, 5.5173426, -6.5569053, 5.5308518, -12.0708904, 12.0742474
6: -6.6403832, 7.0413785, -6.6569500, 7.0587082, -13.6990910, 13.6983271
7: -8.3214378, 4.1806955, -8.3428059, 4.1933599, -12.5147972, 12.5235004
8: -8.1715746, 6.0167332, -8.1921329, 6.0312138, -14.2027884, 14.2088652
9: -6.3204532, 6.7643962, -6.3373275, 6.7819953, -13.1024475, 13.1017227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117112, upper bound: 9.0094184
time: 7.73 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
time: 5.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0136287, upper bound: 9.0097746
time: 4.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.4618921, 8.7861767, -7.5985565, 5.8469539, -17.3088455, 16.3847332
1: -9.2981081, 7.8428025, -6.1161723, 5.2967691, -14.5948753, 13.9589748
2: -11.2832632, 6.5575161, -7.4525409, 4.5020156, -15.7852783, 14.0100574
3: -13.7841749, 6.4289460, -9.0466480, 4.4073906, -18.1915627, 15.4755917
4: -12.3545294, 9.4555264, -8.1858864, 6.3809571, -18.7354832, 17.6414127
5: -9.9948931, 8.1548386, -6.5427670, 5.5195389, -15.5144320, 14.6976032
6: -9.9219246, 10.5157146, -6.6431484, 7.0443134, -16.9662361, 17.1588631
7: -12.3882370, 6.1962557, -8.3251963, 4.1827192, -16.5709553, 14.5214520
8: -12.2955990, 8.8917942, -8.1749477, 6.0191197, -18.3147182, 17.0667419
9: -9.4355602, 10.0889034, -6.3230643, 6.7672729, -16.2028313, 16.4119682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=240, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0090326, upper bound: 9.0081956
time: 2.04 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
time: 2.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -9.0136287, upper bound: 9.0097746
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -9.0090326, upper bound: 9.0081956
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.5923748, 5.8419423, -7.4363236, 5.7155976, -13.3079720, 13.2782660
1: -6.1107802, 5.2922702, -5.9711609, 5.1793494, -11.2901278, 11.2634315
2: -7.4455948, 4.4978485, -7.2749095, 4.3961573, -11.8417521, 11.7727585
3: -9.0388641, 4.4034162, -8.8425083, 4.2966771, -13.3355408, 13.2459240
4: -8.1786327, 6.3753219, -8.0005665, 6.2316737, -14.4103069, 14.3758888
5: -6.5369520, 5.5147843, -6.3874593, 5.3953571, -11.9323092, 11.9022436
6: -6.6373873, 7.0383248, -6.4942737, 6.8905997, -13.5279856, 13.5325985
7: -8.3177099, 4.1779284, -8.1398649, 4.0495753, -12.3672848, 12.3177929
8: -8.1678791, 6.0140491, -7.9943476, 5.8853970, -14.0532761, 14.0083961
9: -6.3169584, 6.7610679, -6.1536164, 6.6015611, -12.9185200, 12.9146824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100879, upper bound: 9.0078425
time: 3.14 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0082852
time: 4.15 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0085468
time: 5.67 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.5884371, 5.8387451, -7.6525245, 5.8821526, -13.4705877, 13.4912682
1: -6.1073198, 5.2894077, -6.1535773, 5.3257508, -11.4330711, 11.4429846
2: -7.4411883, 4.4952965, -7.4937177, 4.5179243, -11.9591122, 11.9890137
3: -9.0338669, 4.4007907, -9.1112919, 4.4179034, -13.4517708, 13.5120831
4: -8.1741524, 6.3717308, -8.2390251, 6.4093227, -14.5834742, 14.6107559
5: -6.5331135, 5.5116839, -6.5827065, 5.5467153, -12.0798244, 12.0943909
6: -6.6337457, 7.0346417, -6.6816645, 7.0888405, -13.7225857, 13.7163048
7: -8.3132515, 4.1746616, -8.3745699, 4.1748061, -12.4880581, 12.5492306
8: -8.1633711, 6.0108495, -8.2279015, 6.0499687, -14.2133398, 14.2387505
9: -6.3128705, 6.7570658, -6.3381491, 6.7948503, -13.1077204, 13.0952129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0114802, upper bound: 9.0092388
time: 6.27 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130703, upper bound: 9.0095842
time: 3.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0136184, upper bound: 9.0097577
time: 5.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.4590921, 8.7839947, -7.4219718, 5.7042665, -17.1633587, 16.2059669
1: -9.2956934, 7.8408446, -5.9588928, 5.1693621, -14.4650545, 13.7997379
2: -11.2803421, 6.5557642, -7.2598996, 4.3873205, -15.6676617, 13.8156643
3: -13.7806768, 6.4271078, -8.8247700, 4.2883329, -18.0690098, 15.2518778
4: -12.3513479, 9.4531269, -7.9841533, 6.2193427, -18.5706863, 17.4372807
5: -9.9922781, 8.1527605, -6.3743463, 5.3849320, -15.3772106, 14.5271072
6: -9.9194155, 10.5131283, -6.4814320, 6.8769960, -16.7964096, 16.9945602
7: -12.3851204, 6.1942167, -8.1233196, 4.0406828, -16.4258041, 14.3175344
8: -12.2925129, 8.8896236, -7.9785438, 5.8741412, -18.1666527, 16.8681679
9: -9.4329290, 10.0862093, -6.1408749, 6.5884418, -16.0213699, 16.2270851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=240, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075662
time: 4.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
time: 5.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.4556694, 8.7813435, -7.6380620, 5.8706594, -17.3263245, 16.4194050
1: -9.2927818, 7.8384976, -6.1410298, 5.3156662, -14.6084480, 13.9795265
2: -11.2768126, 6.5537090, -7.4782314, 4.5089769, -15.7857895, 14.0319395
3: -13.7764349, 6.4249015, -9.0934429, 4.4092674, -18.1857033, 15.5183449
4: -12.3475323, 9.4502487, -8.2225494, 6.3967052, -18.7442379, 17.6727982
5: -9.9890680, 8.1502419, -6.5694218, 5.5362449, -15.5253096, 14.7196636
6: -9.9163971, 10.5100307, -6.6686773, 7.0750532, -16.9914474, 17.1787071
7: -12.3814268, 6.1917791, -8.3578024, 4.1651478, -16.5465736, 14.5495806
8: -12.2887793, 8.8870096, -8.2119179, 6.0383840, -18.3271637, 17.0989265
9: -9.4297972, 10.0829716, -6.3248234, 6.7809925, -16.2107887, 16.4077950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=240, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0091226, upper bound: 9.0090207
time: 5.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0090142, upper bound: 9.0090142
time: 2.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0082852
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0085468
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0130703, upper bound: 9.0095842
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0136184, upper bound: 9.0097577
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075662
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0091226, upper bound: 9.0090207
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.78
Output dim: 7, lower bound: -9.0090142, upper bound: 9.0090142

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.2951283, 5.6041803, -7.3941441, 5.6824331, -12.9775620, 12.9983244
1: -5.8517585, 5.0813813, -5.9351993, 5.1500201, -11.0017786, 11.0165806
2: -7.1281223, 4.3125429, -7.2313142, 4.3704424, -11.4985647, 11.5438576
3: -8.6604109, 4.2116218, -8.7900705, 4.2716918, -12.9321022, 13.0016918
4: -7.8369484, 6.1117806, -7.9519782, 6.1952953, -14.0322437, 14.0637589
5: -6.2597785, 5.2935662, -6.3489695, 5.3646045, -11.6243830, 11.6425362
6: -6.3685064, 6.7573948, -6.4566894, 6.8511291, -13.2196350, 13.2140846
7: -7.9806423, 3.9685974, -8.0925293, 4.0230618, -12.0037041, 12.0611267
8: -7.8402033, 5.7750745, -7.9481344, 5.8525405, -13.6927433, 13.7232094
9: -6.0361071, 6.4765573, -6.1159353, 6.5632925, -12.5993996, 12.5924931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0082852
time: 5.70 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0082852
time: 3.22 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.4671726, 5.7421360, -7.4363236, 5.7155976, -13.1827698, 13.1784592
1: -6.0023098, 5.2053590, -5.9711609, 5.1793494, -11.1816578, 11.1765194
2: -7.3115797, 4.4213462, -7.2749095, 4.3961573, -11.7077360, 11.6962557
3: -8.8853245, 4.3267407, -8.8425083, 4.2966771, -13.1820011, 13.1692486
4: -8.0373325, 6.2653341, -8.0005665, 6.2316737, -14.2690039, 14.2659006
5: -6.4183636, 5.4223051, -6.3874593, 5.3953571, -11.8137197, 11.8097649
6: -6.5258751, 6.9216070, -6.4942737, 6.8905997, -13.4164743, 13.4158802
7: -8.1785431, 4.0865369, -8.1398649, 4.0495753, -12.2281189, 12.2264013
8: -8.0295496, 5.9145718, -7.9943476, 5.8853970, -13.9149456, 13.9089193
9: -6.1979084, 6.6390562, -6.1536164, 6.6015611, -12.7994690, 12.7926722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100826, upper bound: 9.0078287
time: 5.35 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0085468
time: 3.49 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0085468
time: 6.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.2917819, 5.6015844, -7.6079206, 5.8470721, -13.1388540, 13.2095041
1: -5.8488879, 5.0790267, -6.1151690, 5.2948728, -11.1437607, 11.1941948
2: -7.1246719, 4.3105078, -7.4466724, 4.4907994, -11.6154709, 11.7571793
3: -8.6561594, 4.2095280, -9.0562449, 4.3908277, -13.0469847, 13.2657728
4: -7.8332672, 6.1089363, -8.1877918, 6.3705792, -14.2038460, 14.2967281
5: -6.2567415, 5.2911315, -6.5418811, 5.5144639, -11.7712059, 11.8330126
6: -6.3655252, 6.7543035, -6.6419034, 7.0470958, -13.4126205, 13.3962059
7: -7.9769173, 3.9663796, -8.3245306, 4.1446075, -12.1215248, 12.2909107
8: -7.8365612, 5.7724648, -8.1788788, 6.0148253, -13.8513870, 13.9513416
9: -6.0330019, 6.4734626, -6.2973704, 6.7523150, -12.7853165, 12.7708330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0090278, upper bound: 9.0066096
time: 5.96 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0104584, upper bound: 9.0075992
time: 5.68 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0095630
time: 5.98 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0095184
time: 6.47 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.4636807, 5.7393198, -7.6525245, 5.8821526, -13.3458309, 13.3918419
1: -5.9991961, 5.2028408, -6.1535773, 5.3257508, -11.3249474, 11.3564177
2: -7.3077803, 4.4191070, -7.4937177, 4.5179243, -11.8257036, 11.9128246
3: -8.8808861, 4.3244038, -9.1112919, 4.4179034, -13.2987900, 13.4356937
4: -8.0333786, 6.2621484, -8.2390251, 6.4093227, -14.4427004, 14.5011711
5: -6.4151001, 5.4196901, -6.5827065, 5.5467153, -11.9618130, 12.0023947
6: -6.5226803, 6.9182849, -6.6816645, 7.0888405, -13.6115208, 13.5999470
7: -8.1745396, 4.0838637, -8.3745699, 4.1748061, -12.3493462, 12.4584332
8: -8.0256958, 5.9116774, -8.2279015, 6.0499687, -14.0756636, 14.1395769
9: -6.1943388, 6.6355662, -6.3381491, 6.7948503, -12.9891872, 12.9737139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0114706, upper bound: 9.0092243
time: 7.32 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0097460
time: 4.29 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0097443
time: 5.60 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.0533085, 8.4728231, -7.4198213, 5.7025990, -16.7559071, 15.8926449
1: -8.9547663, 7.5661964, -5.9570746, 5.1678867, -14.1226530, 13.5232716
2: -10.8701487, 6.3273549, -7.2577324, 4.3860950, -15.2562437, 13.5850868
3: -13.2700558, 6.1797357, -8.8220329, 4.2870278, -17.5570831, 15.0017681
4: -11.9081030, 9.1207609, -7.9818001, 6.2175517, -18.1256542, 17.1025620
5: -9.6199284, 7.8643222, -6.3723769, 5.3833866, -15.0033150, 14.2366991
6: -9.5676718, 10.1487761, -6.4795518, 6.8750486, -16.4427204, 16.6283283
7: -11.9579077, 5.9389839, -8.1210318, 4.0393600, -15.9972677, 14.0600157
8: -11.8565874, 8.5830193, -7.9762430, 5.8724957, -17.7290821, 16.5592613
9: -9.0826244, 9.7187929, -6.1389980, 6.5865297, -15.6691542, 15.8577909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=239, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075562
time: 2.30 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075657
time: 1.94 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.2464590, 8.6212597, -7.4006720, 5.6877961, -16.9342556, 16.0219307
1: -9.1164093, 7.6961255, -5.9408455, 5.1546602, -14.2710676, 13.6369705
2: -11.0653582, 6.4351859, -7.2383490, 4.3750620, -15.4404202, 13.6735344
3: -13.5093527, 6.2863898, -8.7975254, 4.2752795, -17.7846317, 15.0839138
4: -12.1183882, 9.2779636, -7.9607000, 6.2015009, -18.3198853, 17.2386627
5: -9.7950773, 7.9993744, -6.3549752, 5.3696089, -15.1646862, 14.3543491
6: -9.7342577, 10.3246508, -6.4627323, 6.8576722, -16.5919285, 16.7873840
7: -12.1667547, 6.0509129, -8.1006193, 4.0273948, -16.1941490, 14.1515322
8: -12.0645151, 8.7292757, -7.9556456, 5.8577995, -17.9223137, 16.6849213
9: -9.2456188, 9.8909664, -6.1220460, 6.5693445, -15.8149633, 16.0130119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=239, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075568
time: 3.30 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
time: 2.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.8261642, 8.3009071, -7.6380620, 5.8706594, -16.6968193, 15.9389687
1: -8.7721119, 7.4214058, -6.1410298, 5.3156662, -14.0877762, 13.5624342
2: -10.6480465, 6.2106442, -7.4782314, 4.5089769, -15.1570215, 13.6888742
3: -13.0088406, 6.0932913, -9.0934429, 4.4092674, -17.4181080, 15.1867342
4: -11.6631784, 8.9431324, -8.2225494, 6.3967052, -18.0598831, 17.1656799
5: -9.4242382, 7.7166719, -6.5694218, 5.5362449, -14.9604826, 14.2860937
6: -9.3797398, 9.9421654, -6.6686773, 7.0750532, -16.4547920, 16.6108418
7: -11.7136784, 5.8486576, -8.3578024, 4.1651478, -15.8788233, 14.2064590
8: -11.6127033, 8.4161406, -8.2119179, 6.0383840, -17.6510868, 16.6280575
9: -8.9090204, 9.5334072, -6.3248234, 6.7809925, -15.6900129, 15.8582306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=238, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0085054, upper bound: 9.0083812
time: 2.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0085093, upper bound: 9.0084109
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.8493872, 8.3194418, -7.5665450, 5.8160172, -16.6654053, 15.8859844
1: -8.7922859, 7.4370656, -6.0817170, 5.2680206, -14.0603056, 13.5187817
2: -10.6730623, 6.2256188, -7.4066758, 4.4699020, -15.1429625, 13.6322937
3: -13.0345373, 6.1015472, -9.0055580, 4.3709354, -17.4054699, 15.1071033
4: -11.6900930, 8.9627151, -8.1448746, 6.3388147, -18.0289078, 17.1075897
5: -9.4454641, 7.7330799, -6.5054722, 5.4869394, -14.9324026, 14.2385521
6: -9.4000587, 9.9656868, -6.6074510, 7.0104451, -16.4105034, 16.5731373
7: -11.7429361, 5.8620586, -8.2817936, 4.1260409, -15.8689756, 14.1438522
8: -11.6399784, 8.4341755, -8.1352835, 5.9846072, -17.6245861, 16.5694561
9: -8.9291668, 9.5555773, -6.2653894, 6.7184820, -15.6476488, 15.8209658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=238, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0084922
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0090140
time: 2.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.87 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0082852
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0082852
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0085468
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0085468
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0095630
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0119016, upper bound: 9.0095184
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0097460
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0128017, upper bound: 9.0097443
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075562
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075657
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075568
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0085054, upper bound: 9.0083812
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0085093, upper bound: 9.0084109
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0084922
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.87
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0090140

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.1445918, 5.4870911, -7.3941441, 5.6824331, -12.8270245, 12.8812351
1: -5.7214813, 4.9746404, -5.9351993, 5.1500201, -10.8715019, 10.9098396
2: -6.9726610, 4.2187037, -7.2313142, 4.3704424, -11.3431034, 11.4500179
3: -8.4706030, 4.1189814, -8.7900705, 4.2716918, -12.7422943, 12.9090519
4: -7.6704855, 5.9831052, -7.9519782, 6.1952953, -13.8657808, 13.9350834
5: -6.1220036, 5.1832900, -6.3489695, 5.3646045, -11.4866085, 11.5322590
6: -6.2343211, 6.6169000, -6.4566894, 6.8511291, -13.0854502, 13.0735893
7: -7.8103848, 3.8658605, -8.0925293, 4.0230618, -11.8334465, 11.9583893
8: -7.6746058, 5.6575179, -7.9481344, 5.8525405, -13.5271463, 13.6056519
9: -5.8927612, 6.3356714, -6.1159353, 6.5632925, -12.4560537, 12.4516068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113665, upper bound: 9.0076456
time: 5.29 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113344, upper bound: 9.0076472
time: 3.00 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.3391104, 5.6369953, -7.3941441, 5.6824331, -13.0215435, 13.0311394
1: -5.8852587, 5.1067724, -5.9351993, 5.1500201, -11.0352783, 11.0419712
2: -7.1682444, 4.3278728, -7.2313142, 4.3704424, -11.5386868, 11.5591869
3: -8.7144756, 4.2256660, -8.7900705, 4.2716918, -12.9861679, 13.0157366
4: -7.8829060, 6.1418548, -7.9519782, 6.1952953, -14.0782013, 14.0938330
5: -6.2976155, 5.3195524, -6.3489695, 5.3646045, -11.6622200, 11.6685219
6: -6.4023457, 6.7956538, -6.4566894, 6.8511291, -13.2534752, 13.2523432
7: -8.0222626, 3.9745965, -8.0925293, 4.0230618, -12.0453243, 12.0671253
8: -7.8846130, 5.8053985, -7.9481344, 5.8525405, -13.7371540, 13.7535324
9: -6.0580292, 6.5068078, -6.1159353, 6.5632925, -12.6213217, 12.6227436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113665, upper bound: 9.0076456
time: 4.97 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113344, upper bound: 9.0076472
time: 5.69 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.3071508, 5.6145029, -7.4363236, 5.7155976, -13.0227489, 13.0508270
1: -5.8611116, 5.0896888, -5.9711609, 5.1793494, -11.0404606, 11.0608501
2: -7.1416125, 4.3177915, -7.2749095, 4.3961573, -11.5377693, 11.5927010
3: -8.6820259, 4.2208681, -8.8425083, 4.2966771, -12.9787025, 13.0633764
4: -7.8532271, 6.1210890, -8.0005665, 6.2316737, -14.0849009, 14.1216555
5: -6.2700338, 5.3017583, -6.3874593, 5.3953571, -11.6653910, 11.6892176
6: -6.3796101, 6.7699833, -6.4942737, 6.8905997, -13.2702103, 13.2642574
7: -7.9954062, 3.9688315, -8.1398649, 4.0495753, -12.0449810, 12.1086960
8: -7.8525290, 5.7851710, -7.9943476, 5.8853970, -13.7379265, 13.7795181
9: -6.0381689, 6.4843922, -6.1536164, 6.6015611, -12.6397305, 12.6380081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100715, upper bound: 9.0078191
time: 3.40 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0112890, upper bound: 9.0082459
time: 3.18 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123400, upper bound: 9.0079245
time: 5.07 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123365, upper bound: 9.0079306
time: 4.21 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.5171986, 5.7766552, -7.4363236, 5.7155976, -13.2327957, 13.2129784
1: -6.0381174, 5.2324224, -5.9711609, 5.1793494, -11.2174664, 11.2035828
2: -7.3530979, 4.4363537, -7.2749095, 4.3961573, -11.7492552, 11.7112637
3: -8.9442310, 4.3378239, -8.8425083, 4.2966771, -13.2409077, 13.1803322
4: -8.0849705, 6.2937298, -8.0005665, 6.2316737, -14.3166447, 14.2942963
5: -6.4596601, 5.4494090, -6.3874593, 5.3953571, -11.8550167, 11.8368683
6: -6.5618014, 6.9629698, -6.4942737, 6.8905997, -13.4524012, 13.4572430
7: -8.2238283, 4.0871277, -8.1398649, 4.0495753, -12.2734032, 12.2269926
8: -8.0793037, 5.9449964, -7.9943476, 5.8853970, -13.9647007, 13.9393444
9: -6.2170205, 6.6690388, -6.1536164, 6.6015611, -12.8185816, 12.8226547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=228, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100715, upper bound: 9.0078287
time: 6.45 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0112890, upper bound: 9.0082459
time: 6.43 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123400, upper bound: 9.0079245
time: 5.65 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123365, upper bound: 9.0079306
time: 4.25 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.1445918, 5.4870911, -7.6079206, 5.8470721, -12.9916639, 13.0950117
1: -5.7214813, 4.9746404, -6.1151690, 5.2948728, -11.0163536, 11.0898094
2: -6.9726610, 4.2187037, -7.4466724, 4.4907994, -11.4634609, 11.6653748
3: -8.4706030, 4.1189814, -9.0562449, 4.3908277, -12.8614302, 13.1752262
4: -7.6704855, 5.9831052, -8.1877918, 6.3705792, -14.0410652, 14.1708946
5: -6.1220036, 5.1832900, -6.5418811, 5.5144639, -11.6364670, 11.7251711
6: -6.2343211, 6.6169000, -6.6419034, 7.0470958, -13.2814169, 13.2588034
7: -7.8103848, 3.8658605, -8.3245306, 4.1446075, -11.9549904, 12.1903915
8: -7.6746058, 5.6575179, -8.1788788, 6.0148253, -13.6894312, 13.8363962
9: -5.8927612, 6.3356714, -6.2973704, 6.7523150, -12.6450758, 12.6330414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113665, upper bound: 9.0089267
time: 4.56 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113344, upper bound: 9.0089607
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -7.3391104, 5.6369953, -7.6079206, 5.8470721, -13.1861820, 13.2449160
1: -5.8852587, 5.1067724, -6.1151690, 5.2948728, -11.1801319, 11.2219391
2: -7.1682444, 4.3278728, -7.4466724, 4.4907994, -11.6590443, 11.7745457
3: -8.7144756, 4.2256660, -9.0562449, 4.3908277, -13.1053028, 13.2819109
4: -7.8829060, 6.1418548, -8.1877918, 6.3705792, -14.2534847, 14.3296452
5: -6.2976155, 5.3195524, -6.5418811, 5.5144639, -11.8120794, 11.8614330
6: -6.4023457, 6.7956538, -6.6419034, 7.0470958, -13.4494419, 13.4375572
7: -8.0222626, 3.9745965, -8.3245306, 4.1446075, -12.1668673, 12.2991276
8: -7.8846130, 5.8053985, -8.1788788, 6.0148253, -13.8994389, 13.9842768
9: -6.0580292, 6.5068078, -6.2973704, 6.7523150, -12.8103428, 12.8041782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113665, upper bound: 9.0088771
time: 3.52 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113344, upper bound: 9.0089089
time: 5.58 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -7.3071508, 5.6145029, -7.6525245, 5.8821526, -13.1893034, 13.2670250
1: -5.8611116, 5.0896888, -6.1535773, 5.3257508, -11.1868629, 11.2432642
2: -7.1416125, 4.3177915, -7.4937177, 4.5179243, -11.6595364, 11.8115082
3: -8.6820259, 4.2208681, -9.1112919, 4.4179034, -13.0999298, 13.3321600
4: -7.8532271, 6.1210890, -8.2390251, 6.4093227, -14.2625504, 14.3601112
5: -6.2700338, 5.3017583, -6.5827065, 5.5467153, -11.8167496, 11.8844643
6: -6.3796101, 6.7699833, -6.6816645, 7.0888405, -13.4684505, 13.4516468
7: -7.9954062, 3.9688315, -8.3745699, 4.1748061, -12.1702118, 12.3434010
8: -7.8525290, 5.7851710, -8.2279015, 6.0499687, -13.9024982, 14.0130692
9: -6.0381689, 6.4843922, -6.3381491, 6.7948503, -12.8330183, 12.8225412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100715, upper bound: 9.0090699
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 6.40 + 594.78 = 601.18 seconds
