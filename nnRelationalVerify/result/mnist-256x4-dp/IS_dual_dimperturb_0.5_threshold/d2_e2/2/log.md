## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00285264


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0004590, 0.0021489, -0.0004590, 0.0021489, -0.0026079, 0.0026079)
1: (0.9913285, 0.9968511, 0.9913285, 0.9968511, -0.0055226, 0.0055226)
2: (-0.0082355, -0.0038534, -0.0082355, -0.0038534, -0.0043821, 0.0043821)
3: (0.0021139, 0.0053765, 0.0021139, 0.0053765, -0.0032626, 0.0032626)
4: (0.0014625, 0.0069602, 0.0014625, 0.0069602, -0.0054977, 0.0054977)
5: (0.0022747, 0.0084546, 0.0022747, 0.0084546, -0.0061798, 0.0061798)
6: (-0.0042628, 0.0017236, -0.0042628, 0.0017236, -0.0059864, 0.0059864)
7: (-0.0089185, -0.0062751, -0.0089185, -0.0062751, -0.0026435, 0.0026435)
8: (0.0033145, 0.0082519, 0.0033145, 0.0082519, -0.0049374, 0.0049374)
9: (-0.0051709, -0.0013961, -0.0051709, -0.0013961, -0.0037748, 0.0037748)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.20 + 2.03 = 3.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032993, upper bound: 0.0032993

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032279, upper bound: 0.0031027
time: 0.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032474, upper bound: 0.0032474
time: 1.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 1, lower bound: -0.0032279, upper bound: 0.0031027
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 1, lower bound: -0.0032474, upper bound: 0.0032474

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0002748, 0.0021385, -0.0004351, 0.0021485, -0.0024232, 0.0025736
1: 0.9913506, 0.9964611, 0.9913295, 0.9968005, -0.0054500, 0.0051316
2: -0.0081433, -0.0038894, -0.0082235, -0.0038549, -0.0042884, 0.0043341
3: 0.0023443, 0.0053635, 0.0021438, 0.0053760, -0.0030316, 0.0032197
4: 0.0014910, 0.0069243, 0.0014637, 0.0069587, -0.0054677, 0.0054606
5: 0.0027112, 0.0084298, 0.0023313, 0.0084535, -0.0057423, 0.0060986
6: -0.0042345, 0.0013200, -0.0042616, 0.0016713, -0.0059058, 0.0055816
7: -0.0089080, -0.0064618, -0.0089181, -0.0062993, -0.0026087, 0.0024563
8: 0.0033618, 0.0082273, 0.0033165, 0.0082487, -0.0048869, 0.0049108
9: -0.0051558, -0.0016627, -0.0051702, -0.0014307, -0.0037251, 0.0035075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0031027
time: 1.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0031027
time: 1.20 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0003946, 0.0021472, -0.0004590, 0.0021489, -0.0025436, 0.0026062
1: 0.9913321, 0.9967150, 0.9913285, 0.9968511, -0.0055191, 0.0053865
2: -0.0082033, -0.0038591, -0.0082355, -0.0038534, -0.0043499, 0.0043763
3: 0.0021944, 0.0053744, 0.0021139, 0.0053765, -0.0031822, 0.0032605
4: 0.0014671, 0.0069545, 0.0014625, 0.0069602, -0.0054931, 0.0054919
5: 0.0024271, 0.0084506, 0.0022747, 0.0084546, -0.0060274, 0.0061759
6: -0.0042582, 0.0015827, -0.0042628, 0.0017236, -0.0059818, 0.0058454
7: -0.0089168, -0.0063403, -0.0089185, -0.0062751, -0.0026418, 0.0025783
8: 0.0033221, 0.0082433, 0.0033145, 0.0082519, -0.0049298, 0.0049288
9: -0.0051685, -0.0014892, -0.0051709, -0.0013961, -0.0037724, 0.0036817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0032278
time: 1.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0032474
time: 1.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.36 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0031027
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0031027
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0032278
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 1, lower bound: -0.0031027, upper bound: 0.0032474

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002748, 0.0021385, -0.0002748, 0.0021385, -0.0024132, 0.0024132
1: 0.9913506, 0.9964611, 0.9913506, 0.9964611, -0.0051105, 0.0051105
2: -0.0081433, -0.0038894, -0.0081433, -0.0038894, -0.0042539, 0.0042539
3: 0.0023443, 0.0053635, 0.0023443, 0.0053635, -0.0030191, 0.0030191
4: 0.0014910, 0.0069243, 0.0014910, 0.0069243, -0.0054333, 0.0054333
5: 0.0027112, 0.0084298, 0.0027112, 0.0084298, -0.0057186, 0.0057186
6: -0.0042345, 0.0013200, -0.0042345, 0.0013200, -0.0055545, 0.0055545
7: -0.0089080, -0.0064618, -0.0089080, -0.0064618, -0.0024462, 0.0024462
8: 0.0033618, 0.0082273, 0.0033618, 0.0082273, -0.0048655, 0.0048655
9: -0.0051558, -0.0016627, -0.0051558, -0.0016627, -0.0034931, 0.0034931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030946
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
time: 1.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002748, 0.0021385, -0.0003946, 0.0021472, -0.0024220, 0.0025331
1: 0.9913506, 0.9964611, 0.9913321, 0.9967150, -0.0053644, 0.0051290
2: -0.0081433, -0.0038894, -0.0082033, -0.0038591, -0.0042842, 0.0043139
3: 0.0023443, 0.0053635, 0.0021944, 0.0053744, -0.0030301, 0.0031691
4: 0.0014910, 0.0069243, 0.0014671, 0.0069545, -0.0054635, 0.0054572
5: 0.0027112, 0.0084298, 0.0024271, 0.0084506, -0.0057394, 0.0060027
6: -0.0042345, 0.0013200, -0.0042582, 0.0015827, -0.0058172, 0.0055782
7: -0.0089080, -0.0064618, -0.0089168, -0.0063403, -0.0025677, 0.0024550
8: 0.0033618, 0.0082273, 0.0033221, 0.0082433, -0.0048815, 0.0049052
9: -0.0051558, -0.0016627, -0.0051685, -0.0014892, -0.0036666, 0.0035057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030946
time: 1.81 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
time: 1.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003946, 0.0021472, -0.0002748, 0.0021385, -0.0025331, 0.0024220
1: 0.9913321, 0.9967150, 0.9913506, 0.9964611, -0.0051290, 0.0053644
2: -0.0082033, -0.0038591, -0.0081433, -0.0038894, -0.0043139, 0.0042842
3: 0.0021944, 0.0053744, 0.0023443, 0.0053635, -0.0031691, 0.0030301
4: 0.0014671, 0.0069545, 0.0014910, 0.0069243, -0.0054572, 0.0054635
5: 0.0024271, 0.0084506, 0.0027112, 0.0084298, -0.0060027, 0.0057394
6: -0.0042582, 0.0015827, -0.0042345, 0.0013200, -0.0055782, 0.0058172
7: -0.0089168, -0.0063403, -0.0089080, -0.0064618, -0.0024550, 0.0025677
8: 0.0033221, 0.0082433, 0.0033618, 0.0082273, -0.0049052, 0.0048815
9: -0.0051685, -0.0014892, -0.0051558, -0.0016627, -0.0035057, 0.0036666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030946, upper bound: 0.0031778
time: 1.30 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0031841
time: 1.27 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003946, 0.0021472, -0.0003946, 0.0021472, -0.0025419, 0.0025419
1: 0.9913321, 0.9967150, 0.9913321, 0.9967150, -0.0053830, 0.0053830
2: -0.0082033, -0.0038591, -0.0082033, -0.0038591, -0.0043441, 0.0043441
3: 0.0021944, 0.0053744, 0.0021944, 0.0053744, -0.0031801, 0.0031801
4: 0.0014671, 0.0069545, 0.0014671, 0.0069545, -0.0054874, 0.0054874
5: 0.0024271, 0.0084506, 0.0024271, 0.0084506, -0.0060235, 0.0060235
6: -0.0042582, 0.0015827, -0.0042582, 0.0015827, -0.0058409, 0.0058409
7: -0.0089168, -0.0063403, -0.0089168, -0.0063403, -0.0025766, 0.0025766
8: 0.0033221, 0.0082433, 0.0033221, 0.0082433, -0.0049212, 0.0049212
9: -0.0051685, -0.0014892, -0.0051685, -0.0014892, -0.0036793, 0.0036793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030946, upper bound: 0.0031953
time: 1.16 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0032048
time: 1.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030946
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030946
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030946, upper bound: 0.0031778
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0031841
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030946, upper bound: 0.0031953
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.68
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0032048

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0002748, 0.0021385, -0.0024065, 0.0023391
1: 0.9915078, 0.9964467, 0.9913506, 0.9964611, -0.0049533, 0.0050961
2: -0.0081399, -0.0041456, -0.0081433, -0.0038894, -0.0042505, 0.0039977
3: 0.0023528, 0.0052707, 0.0023443, 0.0053635, -0.0030106, 0.0029263
4: 0.0016935, 0.0066686, 0.0014910, 0.0069243, -0.0052308, 0.0051777
5: 0.0027273, 0.0082541, 0.0027112, 0.0084298, -0.0057025, 0.0055428
6: -0.0040336, 0.0013051, -0.0042345, 0.0013200, -0.0053536, 0.0055396
7: -0.0088328, -0.0064687, -0.0089080, -0.0064618, -0.0023710, 0.0024393
8: 0.0036985, 0.0082264, 0.0033618, 0.0082273, -0.0045288, 0.0048554
9: -0.0050484, -0.0016726, -0.0051558, -0.0016627, -0.0033857, 0.0034832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0002742, 0.0021282, -0.0024667, 0.0023398
1: 0.9915050, 0.9965960, 0.9913723, 0.9964599, -0.0049549, 0.0052236
2: -0.0081752, -0.0041413, -0.0081430, -0.0039248, -0.0042504, 0.0040017
3: 0.0022647, 0.0052722, 0.0023450, 0.0053507, -0.0030860, 0.0029272
4: 0.0016901, 0.0066729, 0.0015189, 0.0068890, -0.0051989, 0.0051540
5: 0.0025603, 0.0082570, 0.0027125, 0.0084056, -0.0058452, 0.0055445
6: -0.0040370, 0.0014595, -0.0042067, 0.0013188, -0.0053558, 0.0056663
7: -0.0088340, -0.0063972, -0.0088976, -0.0064624, -0.0023717, 0.0025003
8: 0.0036928, 0.0082358, 0.0034083, 0.0082272, -0.0045344, 0.0048160
9: -0.0050502, -0.0015705, -0.0051410, -0.0016635, -0.0033867, 0.0035704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0003946, 0.0021472, -0.0024152, 0.0024589
1: 0.9915078, 0.9964467, 0.9913321, 0.9967150, -0.0052072, 0.0051146
2: -0.0081399, -0.0041456, -0.0082033, -0.0038591, -0.0042808, 0.0040577
3: 0.0023528, 0.0052707, 0.0021944, 0.0053744, -0.0030216, 0.0030763
4: 0.0016935, 0.0066686, 0.0014671, 0.0069545, -0.0052610, 0.0052016
5: 0.0027273, 0.0082541, 0.0024271, 0.0084506, -0.0057233, 0.0058269
6: -0.0040336, 0.0013051, -0.0042582, 0.0015827, -0.0056163, 0.0055633
7: -0.0088328, -0.0064687, -0.0089168, -0.0063403, -0.0024925, 0.0024481
8: 0.0036985, 0.0082264, 0.0033221, 0.0082433, -0.0045439, 0.0048946
9: -0.0050484, -0.0016726, -0.0051685, -0.0014892, -0.0035592, 0.0034959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030641, upper bound: 0.0030562
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031545, upper bound: 0.0030742
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0003941, 0.0021367, -0.0024752, 0.0024596
1: 0.9915050, 0.9965960, 0.9913543, 0.9967136, -0.0052086, 0.0052416
2: -0.0081752, -0.0041413, -0.0082030, -0.0038955, -0.0042797, 0.0040617
3: 0.0022647, 0.0052722, 0.0021951, 0.0053613, -0.0030966, 0.0030771
4: 0.0016901, 0.0066729, 0.0014958, 0.0069182, -0.0052281, 0.0051771
5: 0.0025603, 0.0082570, 0.0024285, 0.0084257, -0.0058653, 0.0058285
6: -0.0040370, 0.0014595, -0.0042297, 0.0015814, -0.0056183, 0.0056892
7: -0.0088340, -0.0063972, -0.0089062, -0.0063409, -0.0024931, 0.0025089
8: 0.0036928, 0.0082358, 0.0033698, 0.0082432, -0.0045504, 0.0048535
9: -0.0050502, -0.0015705, -0.0051532, -0.0014901, -0.0035601, 0.0035827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031841, upper bound: 0.0030692
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031841, upper bound: 0.0030692
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0003946, 0.0021472, -0.0002680, 0.0020643, -0.0024589, 0.0024152
1: 0.9913321, 0.9967150, 0.9915078, 0.9964467, -0.0051146, 0.0052072
2: -0.0082033, -0.0038591, -0.0081399, -0.0041456, -0.0040577, 0.0042808
3: 0.0021944, 0.0053744, 0.0023528, 0.0052707, -0.0030763, 0.0030216
4: 0.0014671, 0.0069545, 0.0016935, 0.0066686, -0.0052016, 0.0052610
5: 0.0024271, 0.0084506, 0.0027273, 0.0082541, -0.0058269, 0.0057233
6: -0.0042582, 0.0015827, -0.0040336, 0.0013051, -0.0055633, 0.0056163
7: -0.0089168, -0.0063403, -0.0088328, -0.0064687, -0.0024481, 0.0024925
8: 0.0033221, 0.0082433, 0.0036985, 0.0082264, -0.0048946, 0.0045439
9: -0.0051685, -0.0014892, -0.0050484, -0.0016726, -0.0034959, 0.0035592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030562, upper bound: 0.0030641
time: 1.21 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030742, upper bound: 0.0031545
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0003941, 0.0021367, -0.0003385, 0.0020656, -0.0024596, 0.0024752
1: 0.9913543, 0.9967136, 0.9915050, 0.9965960, -0.0052416, 0.0052086
2: -0.0082030, -0.0038955, -0.0081752, -0.0041413, -0.0040617, 0.0042797
3: 0.0021951, 0.0053613, 0.0022647, 0.0052722, -0.0030771, 0.0030966
4: 0.0014958, 0.0069182, 0.0016901, 0.0066729, -0.0051771, 0.0052281
5: 0.0024285, 0.0084257, 0.0025603, 0.0082570, -0.0058285, 0.0058653
6: -0.0042297, 0.0015814, -0.0040370, 0.0014595, -0.0056892, 0.0056183
7: -0.0089062, -0.0063409, -0.0088340, -0.0063972, -0.0025089, 0.0024931
8: 0.0033698, 0.0082432, 0.0036928, 0.0082358, -0.0048535, 0.0045504
9: -0.0051532, -0.0014901, -0.0050502, -0.0015705, -0.0035827, 0.0035601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0031841
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0031841
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0003946, 0.0021472, -0.0003878, 0.0020732, -0.0024678, 0.0025350
1: 0.9913321, 0.9967150, 0.9914889, 0.9967003, -0.0053682, 0.0052261
2: -0.0082033, -0.0038591, -0.0081998, -0.0041150, -0.0040883, 0.0043407
3: 0.0021944, 0.0053744, 0.0022030, 0.0052818, -0.0030874, 0.0031715
4: 0.0014671, 0.0069545, 0.0016693, 0.0066992, -0.0052321, 0.0052852
5: 0.0024271, 0.0084506, 0.0024435, 0.0082751, -0.0058479, 0.0060071
6: -0.0042582, 0.0015827, -0.0040576, 0.0015676, -0.0058258, 0.0056403
7: -0.0089168, -0.0063403, -0.0088418, -0.0063473, -0.0025695, 0.0025015
8: 0.0033221, 0.0082433, 0.0036583, 0.0082424, -0.0049044, 0.0045845
9: -0.0051685, -0.0014892, -0.0050612, -0.0014992, -0.0036693, 0.0035720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0031953
time: 1.21 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0031953
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0003941, 0.0021367, -0.0004500, 0.0020684, -0.0024624, 0.0025867
1: 0.9913543, 0.9967136, 0.9914991, 0.9968323, -0.0054779, 0.0052146
2: -0.0082030, -0.0038955, -0.0082310, -0.0041315, -0.0040715, 0.0043355
3: 0.0021951, 0.0053613, 0.0021251, 0.0052758, -0.0030807, 0.0032362
4: 0.0014958, 0.0069182, 0.0016823, 0.0066827, -0.0051869, 0.0052358
5: 0.0024285, 0.0084257, 0.0022959, 0.0082637, -0.0058352, 0.0061298
6: -0.0042297, 0.0015814, -0.0040447, 0.0017040, -0.0059337, 0.0056260
7: -0.0089062, -0.0063409, -0.0088369, -0.0062842, -0.0026220, 0.0024960
8: 0.0033698, 0.0082432, 0.0036799, 0.0082507, -0.0048625, 0.0045633
9: -0.0051532, -0.0014901, -0.0050543, -0.0014091, -0.0037442, 0.0035642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0032048
time: 1.44 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0032048
time: 2.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.78 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0030692
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030641, upper bound: 0.0030562
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0031545, upper bound: 0.0030742
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0031841, upper bound: 0.0030692
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0031841, upper bound: 0.0030692
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030562, upper bound: 0.0030641
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030742, upper bound: 0.0031545
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0031841
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0030692, upper bound: 0.0031841
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0031953
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0031953
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0032048
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 1, lower bound: -0.0031311, upper bound: 0.0032048

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0002680, 0.0020643, -0.0023323, 0.0023323
1: 0.9915078, 0.9964467, 0.9915078, 0.9964467, -0.0049389, 0.0049389
2: -0.0081399, -0.0041456, -0.0081399, -0.0041456, -0.0039943, 0.0039943
3: 0.0023528, 0.0052707, 0.0023528, 0.0052707, -0.0029178, 0.0029178
4: 0.0016935, 0.0066686, 0.0016935, 0.0066686, -0.0049751, 0.0049751
5: 0.0027273, 0.0082541, 0.0027273, 0.0082541, -0.0055267, 0.0055267
6: -0.0040336, 0.0013051, -0.0040336, 0.0013051, -0.0053387, 0.0053387
7: -0.0088328, -0.0064687, -0.0088328, -0.0064687, -0.0023641, 0.0023641
8: 0.0036985, 0.0082264, 0.0036985, 0.0082264, -0.0045176, 0.0045176
9: -0.0050484, -0.0016726, -0.0050484, -0.0016726, -0.0033758, 0.0033758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030345, upper bound: 0.0029421
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030746
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0003385, 0.0020656, -0.0023335, 0.0024028
1: 0.9915078, 0.9964467, 0.9915050, 0.9965960, -0.0050882, 0.0049416
2: -0.0081399, -0.0041456, -0.0081752, -0.0041413, -0.0039986, 0.0040296
3: 0.0023528, 0.0052707, 0.0022647, 0.0052722, -0.0029194, 0.0030060
4: 0.0016935, 0.0066686, 0.0016901, 0.0066729, -0.0049794, 0.0049786
5: 0.0027273, 0.0082541, 0.0025603, 0.0082570, -0.0055297, 0.0056937
6: -0.0040336, 0.0013051, -0.0040370, 0.0014595, -0.0054931, 0.0053420
7: -0.0088328, -0.0064687, -0.0088340, -0.0063972, -0.0024355, 0.0023653
8: 0.0036985, 0.0082264, 0.0036928, 0.0082358, -0.0045243, 0.0045296
9: -0.0050484, -0.0016726, -0.0050502, -0.0015705, -0.0034779, 0.0033777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0030563
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030746
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0002680, 0.0020643, -0.0024028, 0.0023335
1: 0.9915050, 0.9965960, 0.9915078, 0.9964467, -0.0049416, 0.0050882
2: -0.0081752, -0.0041413, -0.0081399, -0.0041456, -0.0040296, 0.0039986
3: 0.0022647, 0.0052722, 0.0023528, 0.0052707, -0.0030060, 0.0029194
4: 0.0016901, 0.0066729, 0.0016935, 0.0066686, -0.0049786, 0.0049794
5: 0.0025603, 0.0082570, 0.0027273, 0.0082541, -0.0056937, 0.0055297
6: -0.0040370, 0.0014595, -0.0040336, 0.0013051, -0.0053420, 0.0054931
7: -0.0088340, -0.0063972, -0.0088328, -0.0064687, -0.0023653, 0.0024355
8: 0.0036928, 0.0082358, 0.0036985, 0.0082264, -0.0045296, 0.0045243
9: -0.0050502, -0.0015705, -0.0050484, -0.0016726, -0.0033777, 0.0034779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030370, upper bound: 0.0029169
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030458
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0003385, 0.0020656, -0.0024040, 0.0024040
1: 0.9915050, 0.9965960, 0.9915050, 0.9965960, -0.0050910, 0.0050910
2: -0.0081752, -0.0041413, -0.0081752, -0.0041413, -0.0040339, 0.0040339
3: 0.0022647, 0.0052722, 0.0022647, 0.0052722, -0.0030075, 0.0030075
4: 0.0016901, 0.0066729, 0.0016901, 0.0066729, -0.0049829, 0.0049829
5: 0.0025603, 0.0082570, 0.0025603, 0.0082570, -0.0056967, 0.0056967
6: -0.0040370, 0.0014595, -0.0040370, 0.0014595, -0.0054965, 0.0054965
7: -0.0088340, -0.0063972, -0.0088340, -0.0063972, -0.0024368, 0.0024368
8: 0.0036928, 0.0082358, 0.0036928, 0.0082358, -0.0045271, 0.0045271
9: -0.0050502, -0.0015705, -0.0050502, -0.0015705, -0.0034797, 0.0034797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030370, upper bound: 0.0029169
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030458
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002546, 0.0020637, -0.0002989, 0.0021437, -0.0023982, 0.0023626
1: 0.9915090, 0.9964183, 0.9913397, 0.9965124, -0.0050033, 0.0050786
2: -0.0081332, -0.0041478, -0.0081554, -0.0038715, -0.0042617, 0.0040076
3: 0.0023696, 0.0052699, 0.0023141, 0.0053700, -0.0030004, 0.0029558
4: 0.0016952, 0.0066664, 0.0014768, 0.0069422, -0.0052470, 0.0051896
5: 0.0027591, 0.0082526, 0.0026539, 0.0084421, -0.0056830, 0.0055986
6: -0.0040319, 0.0012757, -0.0042485, 0.0013730, -0.0054049, 0.0055243
7: -0.0088321, -0.0064823, -0.0089132, -0.0064373, -0.0023948, 0.0024309
8: 0.0037013, 0.0082246, 0.0033382, 0.0082305, -0.0045292, 0.0048767
9: -0.0050475, -0.0016920, -0.0051633, -0.0016277, -0.0034198, 0.0034713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030641, upper bound: 0.0030562
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030641, upper bound: 0.0030562
time: 1.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0003670, 0.0021442, -0.0024122, 0.0024313
1: 0.9915078, 0.9964467, 0.9913385, 0.9966564, -0.0051486, 0.0051082
2: -0.0081399, -0.0041456, -0.0081894, -0.0038696, -0.0042703, 0.0040438
3: 0.0023528, 0.0052707, 0.0022290, 0.0053706, -0.0030178, 0.0030417
4: 0.0016935, 0.0066686, 0.0014754, 0.0069440, -0.0052505, 0.0051933
5: 0.0027273, 0.0082541, 0.0024928, 0.0084434, -0.0057161, 0.0057613
6: -0.0040336, 0.0013051, -0.0042500, 0.0015220, -0.0055556, 0.0055551
7: -0.0088328, -0.0064687, -0.0089138, -0.0063684, -0.0024644, 0.0024451
8: 0.0036985, 0.0082264, 0.0033358, 0.0082396, -0.0045381, 0.0048808
9: -0.0050484, -0.0016726, -0.0051641, -0.0015293, -0.0035191, 0.0034915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031545, upper bound: 0.0030742
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031545, upper bound: 0.0030742
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0003878, 0.0020732, -0.0024116, 0.0024533
1: 0.9915050, 0.9965960, 0.9914889, 0.9967003, -0.0051953, 0.0051070
2: -0.0081752, -0.0041413, -0.0081998, -0.0041150, -0.0040602, 0.0040585
3: 0.0022647, 0.0052722, 0.0022030, 0.0052818, -0.0030171, 0.0030693
4: 0.0016901, 0.0066729, 0.0016693, 0.0066992, -0.0050091, 0.0050036
5: 0.0025603, 0.0082570, 0.0024435, 0.0082751, -0.0057147, 0.0058135
6: -0.0040370, 0.0014595, -0.0040576, 0.0015676, -0.0056045, 0.0055171
7: -0.0088340, -0.0063972, -0.0088418, -0.0063473, -0.0024867, 0.0024445
8: 0.0036928, 0.0082358, 0.0036583, 0.0082424, -0.0045396, 0.0045645
9: -0.0050502, -0.0015705, -0.0050612, -0.0014992, -0.0035510, 0.0034907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030621, upper bound: 0.0030345
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031609, upper bound: 0.0030458
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0004500, 0.0020684, -0.0024068, 0.0025156
1: 0.9915050, 0.9965960, 0.9914991, 0.9968323, -0.0053272, 0.0050969
2: -0.0081752, -0.0041413, -0.0082310, -0.0041315, -0.0040437, 0.0040897
3: 0.0022647, 0.0052722, 0.0021251, 0.0052758, -0.0030111, 0.0031471
4: 0.0016901, 0.0066729, 0.0016823, 0.0066827, -0.0049927, 0.0049906
5: 0.0025603, 0.0082570, 0.0022959, 0.0082637, -0.0057034, 0.0059611
6: -0.0040370, 0.0014595, -0.0040447, 0.0017040, -0.0057410, 0.0055042
7: -0.0088340, -0.0063972, -0.0088369, -0.0062842, -0.0025499, 0.0024397
8: 0.0036928, 0.0082358, 0.0036799, 0.0082507, -0.0045373, 0.0045398
9: -0.0050502, -0.0015705, -0.0050543, -0.0014091, -0.0036412, 0.0034838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030621, upper bound: 0.0030344
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031609, upper bound: 0.0030458
time: 1.72 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002989, 0.0021437, -0.0002546, 0.0020637, -0.0023626, 0.0023982
1: 0.9913397, 0.9965124, 0.9915090, 0.9964183, -0.0050786, 0.0050033
2: -0.0081554, -0.0038715, -0.0081332, -0.0041478, -0.0040076, 0.0042617
3: 0.0023141, 0.0053700, 0.0023696, 0.0052699, -0.0029558, 0.0030004
4: 0.0014768, 0.0069422, 0.0016952, 0.0066664, -0.0051896, 0.0052470
5: 0.0026539, 0.0084421, 0.0027591, 0.0082526, -0.0055986, 0.0056830
6: -0.0042485, 0.0013730, -0.0040319, 0.0012757, -0.0055243, 0.0054049
7: -0.0089132, -0.0064373, -0.0088321, -0.0064823, -0.0024309, 0.0023948
8: 0.0033382, 0.0082305, 0.0037013, 0.0082246, -0.0048767, 0.0045292
9: -0.0051633, -0.0016277, -0.0050475, -0.0016920, -0.0034713, 0.0034198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030562, upper bound: 0.0030641
time: 1.33 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030562, upper bound: 0.0030641
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003670, 0.0021442, -0.0002680, 0.0020643, -0.0024313, 0.0024122
1: 0.9913385, 0.9966564, 0.9915078, 0.9964467, -0.0051082, 0.0051486
2: -0.0081894, -0.0038696, -0.0081399, -0.0041456, -0.0040438, 0.0042703
3: 0.0022290, 0.0053706, 0.0023528, 0.0052707, -0.0030417, 0.0030178
4: 0.0014754, 0.0069440, 0.0016935, 0.0066686, -0.0051933, 0.0052505
5: 0.0024928, 0.0084434, 0.0027273, 0.0082541, -0.0057613, 0.0057161
6: -0.0042500, 0.0015220, -0.0040336, 0.0013051, -0.0055551, 0.0055556
7: -0.0089138, -0.0063684, -0.0088328, -0.0064687, -0.0024451, 0.0024644
8: 0.0033358, 0.0082396, 0.0036985, 0.0082264, -0.0048808, 0.0045381
9: -0.0051641, -0.0015293, -0.0050484, -0.0016726, -0.0034915, 0.0035191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030742, upper bound: 0.0031545
time: 1.64 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030742, upper bound: 0.0031545
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003878, 0.0020732, -0.0003385, 0.0020656, -0.0024533, 0.0024116
1: 0.9914889, 0.9967003, 0.9915050, 0.9965960, -0.0051070, 0.0051953
2: -0.0081998, -0.0041150, -0.0081752, -0.0041413, -0.0040585, 0.0040602
3: 0.0022030, 0.0052818, 0.0022647, 0.0052722, -0.0030693, 0.0030171
4: 0.0016693, 0.0066992, 0.0016901, 0.0066729, -0.0050036, 0.0050091
5: 0.0024435, 0.0082751, 0.0025603, 0.0082570, -0.0058135, 0.0057147
6: -0.0040576, 0.0015676, -0.0040370, 0.0014595, -0.0055171, 0.0056045
7: -0.0088418, -0.0063473, -0.0088340, -0.0063972, -0.0024445, 0.0024867
8: 0.0036583, 0.0082424, 0.0036928, 0.0082358, -0.0045645, 0.0045396
9: -0.0050612, -0.0014992, -0.0050502, -0.0015705, -0.0034907, 0.0035510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030345, upper bound: 0.0030621
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0031609
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004500, 0.0020684, -0.0003385, 0.0020656, -0.0025156, 0.0024068
1: 0.9914991, 0.9968323, 0.9915050, 0.9965960, -0.0050969, 0.0053272
2: -0.0082310, -0.0041315, -0.0081752, -0.0041413, -0.0040897, 0.0040437
3: 0.0021251, 0.0052758, 0.0022647, 0.0052722, -0.0031471, 0.0030111
4: 0.0016823, 0.0066827, 0.0016901, 0.0066729, -0.0049906, 0.0049927
5: 0.0022959, 0.0082637, 0.0025603, 0.0082570, -0.0059611, 0.0057034
6: -0.0040447, 0.0017040, -0.0040370, 0.0014595, -0.0055042, 0.0057410
7: -0.0088369, -0.0062842, -0.0088340, -0.0063972, -0.0024397, 0.0025499
8: 0.0036799, 0.0082507, 0.0036928, 0.0082358, -0.0045398, 0.0045373
9: -0.0050543, -0.0014091, -0.0050502, -0.0015705, -0.0034838, 0.0036412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030345, upper bound: 0.0030621
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0031545
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003878, 0.0020732, -0.0003878, 0.0020732, -0.0024609, 0.0024609
1: 0.9914889, 0.9967003, 0.9914889, 0.9967003, -0.0052114, 0.0052114
2: -0.0081998, -0.0041150, -0.0081998, -0.0041150, -0.0040848, 0.0040848
3: 0.0022030, 0.0052818, 0.0022030, 0.0052818, -0.0030788, 0.0030788
4: 0.0016693, 0.0066992, 0.0016693, 0.0066992, -0.0050299, 0.0050299
5: 0.0024435, 0.0082751, 0.0024435, 0.0082751, -0.0058316, 0.0058316
6: -0.0040576, 0.0015676, -0.0040576, 0.0015676, -0.0056251, 0.0056251
7: -0.0088418, -0.0063473, -0.0088418, -0.0063473, -0.0024945, 0.0024945
8: 0.0036583, 0.0082424, 0.0036583, 0.0082424, -0.0045675, 0.0045675
9: -0.0050612, -0.0014992, -0.0050612, -0.0014992, -0.0035620, 0.0035620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031055, upper bound: 0.0030797
time: 1.28 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031126, upper bound: 0.0031717
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004500, 0.0020684, -0.0003878, 0.0020732, -0.0025232, 0.0024561
1: 0.9914991, 0.9968323, 0.9914889, 0.9967003, -0.0052012, 0.0053433
2: -0.0082310, -0.0041315, -0.0081998, -0.0041150, -0.0041160, 0.0040683
3: 0.0021251, 0.0052758, 0.0022030, 0.0052818, -0.0031567, 0.0030728
4: 0.0016823, 0.0066827, 0.0016693, 0.0066992, -0.0050168, 0.0050134
5: 0.0022959, 0.0082637, 0.0024435, 0.0082751, -0.0059792, 0.0058203
6: -0.0040447, 0.0017040, -0.0040576, 0.0015676, -0.0056122, 0.0057616
7: -0.0088369, -0.0062842, -0.0088418, -0.0063473, -0.0024896, 0.0025576
8: 0.0036799, 0.0082507, 0.0036583, 0.0082424, -0.0045512, 0.0045732
9: -0.0050543, -0.0014091, -0.0050612, -0.0014992, -0.0035551, 0.0036522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031055, upper bound: 0.0030797
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031126, upper bound: 0.0031717
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003878, 0.0020732, -0.0004500, 0.0020684, -0.0024561, 0.0025232
1: 0.9914889, 0.9967003, 0.9914991, 0.9968323, -0.0053433, 0.0052012
2: -0.0081998, -0.0041150, -0.0082310, -0.0041315, -0.0040683, 0.0041160
3: 0.0022030, 0.0052818, 0.0021251, 0.0052758, -0.0030728, 0.0031567
4: 0.0016693, 0.0066992, 0.0016823, 0.0066827, -0.0050134, 0.0050168
5: 0.0024435, 0.0082751, 0.0022959, 0.0082637, -0.0058203, 0.0059792
6: -0.0040576, 0.0015676, -0.0040447, 0.0017040, -0.0057616, 0.0056122
7: -0.0088418, -0.0063473, -0.0088369, -0.0062842, -0.0025576, 0.0024896
8: 0.0036583, 0.0082424, 0.0036799, 0.0082507, -0.0045732, 0.0045512
9: -0.0050612, -0.0014992, -0.0050543, -0.0014091, -0.0036522, 0.0035551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030456, upper bound: 0.0031713
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031064, upper bound: 0.0031808
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004500, 0.0020684, -0.0004500, 0.0020684, -0.0025184, 0.0025184
1: 0.9914991, 0.9968323, 0.9914991, 0.9968323, -0.0053332, 0.0053332
2: -0.0082310, -0.0041315, -0.0082310, -0.0041315, -0.0040995, 0.0040995
3: 0.0021251, 0.0052758, 0.0021251, 0.0052758, -0.0031507, 0.0031507
4: 0.0016823, 0.0066827, 0.0016823, 0.0066827, -0.0050004, 0.0050004
5: 0.0022959, 0.0082637, 0.0022959, 0.0082637, -0.0059678, 0.0059678
6: -0.0040447, 0.0017040, -0.0040447, 0.0017040, -0.0057487, 0.0057487
7: -0.0088369, -0.0062842, -0.0088369, -0.0062842, -0.0025528, 0.0025528
8: 0.0036799, 0.0082507, 0.0036799, 0.0082507, -0.0045473, 0.0045473
9: -0.0050543, -0.0014091, -0.0050543, -0.0014091, -0.0036453, 0.0036453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030988, upper bound: 0.0030797
time: 1.38 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031064, upper bound: 0.0031717
time: 1.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.10 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030345, upper bound: 0.0029421
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030746
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0030563
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030746
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030370, upper bound: 0.0029169
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030458
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030370, upper bound: 0.0029169
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0030458
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030641, upper bound: 0.0030562
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030641, upper bound: 0.0030562
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031545, upper bound: 0.0030742
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031545, upper bound: 0.0030742
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030621, upper bound: 0.0030345
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031609, upper bound: 0.0030458
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030621, upper bound: 0.0030344
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031609, upper bound: 0.0030458
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030562, upper bound: 0.0030641
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030562, upper bound: 0.0030641
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030742, upper bound: 0.0031545
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030742, upper bound: 0.0031545
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030345, upper bound: 0.0030621
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0031609
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030345, upper bound: 0.0030621
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030458, upper bound: 0.0031545
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031055, upper bound: 0.0030797
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031126, upper bound: 0.0031717
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031055, upper bound: 0.0030797
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031126, upper bound: 0.0031717
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030456, upper bound: 0.0031713
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031064, upper bound: 0.0031808
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0030988, upper bound: 0.0030797
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 1, lower bound: -0.0031064, upper bound: 0.0031717

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001691, 0.0020595, -0.0002546, 0.0020637, -0.0022328, 0.0023141
1: 0.9915178, 0.9962372, 0.9915090, 0.9964183, -0.0049005, 0.0047282
2: -0.0080905, -0.0041621, -0.0081332, -0.0041478, -0.0039427, 0.0039711
3: 0.0024765, 0.0052647, 0.0023696, 0.0052699, -0.0027934, 0.0028951
4: 0.0017065, 0.0066522, 0.0016952, 0.0066664, -0.0049600, 0.0049570
5: 0.0029616, 0.0082428, 0.0027591, 0.0082526, -0.0052909, 0.0054837
6: -0.0040207, 0.0010884, -0.0040319, 0.0012757, -0.0052964, 0.0051203
7: -0.0088279, -0.0065689, -0.0088321, -0.0064823, -0.0023457, 0.0022632
8: 0.0037201, 0.0082132, 0.0037013, 0.0082246, -0.0044944, 0.0045054
9: -0.0050415, -0.0018157, -0.0050475, -0.0016920, -0.0033496, 0.0032318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029895, upper bound: 0.0028433
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030506, upper bound: 0.0029421
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002414, 0.0020614, -0.0002680, 0.0020643, -0.0023057, 0.0023294
1: 0.9915138, 0.9963905, 0.9915078, 0.9964467, -0.0049329, 0.0048827
2: -0.0081267, -0.0041555, -0.0081399, -0.0041456, -0.0039810, 0.0039844
3: 0.0023860, 0.0052671, 0.0023528, 0.0052707, -0.0028846, 0.0029143
4: 0.0017013, 0.0066588, 0.0016935, 0.0066686, -0.0049673, 0.0049653
5: 0.0027902, 0.0082473, 0.0027273, 0.0082541, -0.0054638, 0.0055199
6: -0.0040258, 0.0012469, -0.0040336, 0.0013051, -0.0053309, 0.0052805
7: -0.0088299, -0.0064956, -0.0088328, -0.0064687, -0.0023612, 0.0023372
8: 0.0037115, 0.0082228, 0.0036985, 0.0082264, -0.0045047, 0.0045115
9: -0.0050443, -0.0017110, -0.0050484, -0.0016726, -0.0033717, 0.0033374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030268, upper bound: 0.0030382
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030710, upper bound: 0.0030710
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002546, 0.0020637, -0.0002356, 0.0020583, -0.0023128, 0.0022992
1: 0.9915090, 0.9964183, 0.9915205, 0.9963781, -0.0048690, 0.0048978
2: -0.0081332, -0.0041478, -0.0081237, -0.0041664, -0.0039668, 0.0039759
3: 0.0023696, 0.0052699, 0.0023934, 0.0052631, -0.0028935, 0.0028765
4: 0.0016952, 0.0066664, 0.0017099, 0.0066479, -0.0049527, 0.0049565
5: 0.0027591, 0.0082526, 0.0028041, 0.0082398, -0.0054807, 0.0054485
6: -0.0040319, 0.0012757, -0.0040173, 0.0012341, -0.0052660, 0.0052930
7: -0.0088321, -0.0064823, -0.0088267, -0.0065015, -0.0023306, 0.0023444
8: 0.0037013, 0.0082246, 0.0037258, 0.0082221, -0.0045113, 0.0044930
9: -0.0050475, -0.0016920, -0.0050397, -0.0017195, -0.0033280, 0.0033477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0029211
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0030468
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0003142, 0.0020625, -0.0023305, 0.0023785
1: 0.9915078, 0.9964467, 0.9915115, 0.9965447, -0.0050370, 0.0049351
2: -0.0081399, -0.0041456, -0.0081631, -0.0041519, -0.0039880, 0.0040174
3: 0.0023528, 0.0052707, 0.0022950, 0.0052684, -0.0029156, 0.0029757
4: 0.0016935, 0.0066686, 0.0016985, 0.0066624, -0.0049689, 0.0049702
5: 0.0027273, 0.0082541, 0.0026177, 0.0082497, -0.0055224, 0.0056364
6: -0.0040336, 0.0013051, -0.0040286, 0.0014065, -0.0054401, 0.0053337
7: -0.0088328, -0.0064687, -0.0088309, -0.0064218, -0.0024110, 0.0023622
8: 0.0036985, 0.0082264, 0.0037067, 0.0082326, -0.0045179, 0.0045157
9: -0.0050484, -0.0016726, -0.0050458, -0.0016056, -0.0034428, 0.0033732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0029593
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030440, upper bound: 0.0030626
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002356, 0.0020583, -0.0002546, 0.0020637, -0.0022992, 0.0023128
1: 0.9915205, 0.9963781, 0.9915090, 0.9964183, -0.0048978, 0.0048690
2: -0.0081237, -0.0041664, -0.0081332, -0.0041478, -0.0039759, 0.0039668
3: 0.0023934, 0.0052631, 0.0023696, 0.0052699, -0.0028765, 0.0028935
4: 0.0017099, 0.0066479, 0.0016952, 0.0066664, -0.0049565, 0.0049527
5: 0.0028041, 0.0082398, 0.0027591, 0.0082526, -0.0054485, 0.0054807
6: -0.0040173, 0.0012341, -0.0040319, 0.0012757, -0.0052930, 0.0052660
7: -0.0088267, -0.0065015, -0.0088321, -0.0064823, -0.0023444, 0.0023306
8: 0.0037258, 0.0082221, 0.0037013, 0.0082246, -0.0044930, 0.0045113
9: -0.0050397, -0.0017195, -0.0050475, -0.0016920, -0.0033477, 0.0033280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029211, upper bound: 0.0026852
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030468, upper bound: 0.0029169
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003142, 0.0020625, -0.0002680, 0.0020643, -0.0023785, 0.0023305
1: 0.9915115, 0.9965447, 0.9915078, 0.9964467, -0.0049351, 0.0050370
2: -0.0081631, -0.0041519, -0.0081399, -0.0041456, -0.0040174, 0.0039880
3: 0.0022950, 0.0052684, 0.0023528, 0.0052707, -0.0029757, 0.0029156
4: 0.0016985, 0.0066624, 0.0016935, 0.0066686, -0.0049702, 0.0049689
5: 0.0026177, 0.0082497, 0.0027273, 0.0082541, -0.0056364, 0.0055224
6: -0.0040286, 0.0014065, -0.0040336, 0.0013051, -0.0053337, 0.0054401
7: -0.0088309, -0.0064218, -0.0088328, -0.0064687, -0.0023622, 0.0024110
8: 0.0037067, 0.0082326, 0.0036985, 0.0082264, -0.0045157, 0.0045179
9: -0.0050458, -0.0016056, -0.0050484, -0.0016726, -0.0033732, 0.0034428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029593, upper bound: 0.0028708
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030626, upper bound: 0.0030441
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002356, 0.0020583, -0.0003246, 0.0020649, -0.0023005, 0.0023829
1: 0.9915205, 0.9963781, 0.9915064, 0.9965667, -0.0050462, 0.0048717
2: -0.0081237, -0.0041664, -0.0081682, -0.0041435, -0.0039803, 0.0040018
3: 0.0023934, 0.0052631, 0.0022820, 0.0052715, -0.0028781, 0.0029811
4: 0.0017099, 0.0066479, 0.0016918, 0.0066708, -0.0049609, 0.0049561
5: 0.0028041, 0.0082398, 0.0025932, 0.0082556, -0.0054515, 0.0056466
6: -0.0040173, 0.0012341, -0.0040353, 0.0014292, -0.0054465, 0.0052694
7: -0.0088267, -0.0065015, -0.0088334, -0.0064113, -0.0024154, 0.0023319
8: 0.0037258, 0.0082221, 0.0036956, 0.0082339, -0.0044911, 0.0045148
9: -0.0050397, -0.0017195, -0.0050493, -0.0015906, -0.0034491, 0.0033299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029047, upper bound: 0.0026852
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030317, upper bound: 0.0029169
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003142, 0.0020625, -0.0003385, 0.0020656, -0.0023798, 0.0024009
1: 0.9915115, 0.9965447, 0.9915050, 0.9965960, -0.0050845, 0.0050397
2: -0.0081631, -0.0041519, -0.0081752, -0.0041413, -0.0040218, 0.0040233
3: 0.0022950, 0.0052684, 0.0022647, 0.0052722, -0.0029773, 0.0030037
4: 0.0016985, 0.0066624, 0.0016901, 0.0066729, -0.0049745, 0.0049723
5: 0.0026177, 0.0082497, 0.0025603, 0.0082570, -0.0056393, 0.0056894
6: -0.0040286, 0.0014065, -0.0040370, 0.0014595, -0.0054882, 0.0054434
7: -0.0088309, -0.0064218, -0.0088340, -0.0063972, -0.0024337, 0.0024122
8: 0.0037067, 0.0082326, 0.0036928, 0.0082358, -0.0045132, 0.0045207
9: -0.0050458, -0.0016056, -0.0050502, -0.0015705, -0.0034752, 0.0034446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029410, upper bound: 0.0028708
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030449, upper bound: 0.0030437
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002546, 0.0020637, -0.0002921, 0.0020715, -0.0023261, 0.0023557
1: 0.9915090, 0.9964183, 0.9914925, 0.9964977, -0.0049887, 0.0049258
2: -0.0081332, -0.0041478, -0.0081520, -0.0041208, -0.0040124, 0.0040042
3: 0.0023696, 0.0052699, 0.0023227, 0.0052797, -0.0029101, 0.0029472
4: 0.0016952, 0.0066664, 0.0016739, 0.0066934, -0.0049982, 0.0049926
5: 0.0027591, 0.0082526, 0.0026702, 0.0082711, -0.0055120, 0.0055824
6: -0.0040319, 0.0012757, -0.0040530, 0.0013579, -0.0053898, 0.0053288
7: -0.0088321, -0.0064823, -0.0088401, -0.0064442, -0.0023879, 0.0023578
8: 0.0037013, 0.0082246, 0.0036658, 0.0082296, -0.0045156, 0.0045488
9: -0.0050475, -0.0016920, -0.0050588, -0.0016376, -0.0034099, 0.0033669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029610, upper bound: 0.0029916
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0030455
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002546, 0.0020637, -0.0003490, 0.0020642, -0.0023187, 0.0024126
1: 0.9915090, 0.9964183, 0.9915080, 0.9966183, -0.0051093, 0.0049103
2: -0.0081332, -0.0041478, -0.0081804, -0.0041461, -0.0039872, 0.0040326
3: 0.0023696, 0.0052699, 0.0022515, 0.0052705, -0.0029009, 0.0030184
4: 0.0016952, 0.0066664, 0.0016938, 0.0066682, -0.0049730, 0.0049726
5: 0.0027591, 0.0082526, 0.0025354, 0.0082538, -0.0054947, 0.0057172
6: -0.0040319, 0.0012757, -0.0040333, 0.0014826, -0.0055145, 0.0053090
7: -0.0088321, -0.0064823, -0.0088326, -0.0063866, -0.0024455, 0.0023504
8: 0.0037013, 0.0082246, 0.0036990, 0.0082372, -0.0045205, 0.0045187
9: -0.0050475, -0.0016920, -0.0050482, -0.0015553, -0.0034922, 0.0033563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029610, upper bound: 0.0029916
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0030460
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0003601, 0.0020702, -0.0023382, 0.0024244
1: 0.9915078, 0.9964467, 0.9914953, 0.9966418, -0.0051340, 0.0049514
2: -0.0081399, -0.0041456, -0.0081860, -0.0041252, -0.0040147, 0.0040404
3: 0.0023528, 0.0052707, 0.0022376, 0.0052781, -0.0029252, 0.0030331
4: 0.0016935, 0.0066686, 0.0016774, 0.0066890, -0.0049955, 0.0049913
5: 0.0027273, 0.0082541, 0.0025090, 0.0082681, -0.0055407, 0.0057450
6: -0.0040336, 0.0013051, -0.0040496, 0.0015070, -0.0055405, 0.0053547
7: -0.0088328, -0.0064687, -0.0088388, -0.0063753, -0.0024575, 0.0023701
8: 0.0036985, 0.0082264, 0.0036716, 0.0082387, -0.0045216, 0.0045445
9: -0.0050484, -0.0016726, -0.0050570, -0.0015392, -0.0035092, 0.0033844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030722, upper bound: 0.0030166
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031344, upper bound: 0.0030595
time: 1.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0004250, 0.0020652, -0.0023331, 0.0024893
1: 0.9915078, 0.9964467, 0.9915059, 0.9967793, -0.0052715, 0.0049407
2: -0.0081399, -0.0041456, -0.0082185, -0.0041426, -0.0039973, 0.0040729
3: 0.0023528, 0.0052707, 0.0021563, 0.0052718, -0.0029189, 0.0031143
4: 0.0016935, 0.0066686, 0.0016911, 0.0066717, -0.0049782, 0.0049775
5: 0.0027273, 0.0082541, 0.0023552, 0.0082561, -0.0055288, 0.0058989
6: -0.0040336, 0.0013051, -0.0040359, 0.0016492, -0.0056828, 0.0053410
7: -0.0088328, -0.0064687, -0.0088337, -0.0063095, -0.0025233, 0.0023650
8: 0.0036985, 0.0082264, 0.0036945, 0.0082473, -0.0045268, 0.0045268
9: -0.0050484, -0.0016726, -0.0050497, -0.0014452, -0.0036032, 0.0033771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030722, upper bound: 0.0030166
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031344, upper bound: 0.0030605
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0003246, 0.0020649, -0.0002921, 0.0020715, -0.0023961, 0.0023570
1: 0.9915064, 0.9965667, 0.9914925, 0.9964977, -0.0049913, 0.0050742
2: -0.0081682, -0.0041435, -0.0081520, -0.0041208, -0.0040474, 0.0040085
3: 0.0022820, 0.0052715, 0.0023227, 0.0052797, -0.0029977, 0.0029488
4: 0.0016918, 0.0066708, 0.0016739, 0.0066934, -0.0050016, 0.0049969
5: 0.0025932, 0.0082556, 0.0026702, 0.0082711, -0.0056779, 0.0055853
6: -0.0040353, 0.0014292, -0.0040530, 0.0013579, -0.0053932, 0.0054822
7: -0.0088334, -0.0064113, -0.0088401, -0.0064442, -0.0023892, 0.0024288
8: 0.0036956, 0.0082339, 0.0036658, 0.0082296, -0.0045276, 0.0045554
9: -0.0050493, -0.0015906, -0.0050588, -0.0016376, -0.0034117, 0.0034682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029441, upper bound: 0.0029259
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030575, upper bound: 0.0030265
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0003601, 0.0020702, -0.0024087, 0.0024257
1: 0.9915050, 0.9965960, 0.9914953, 0.9966418, -0.0051368, 0.0051007
2: -0.0081752, -0.0041413, -0.0081860, -0.0041252, -0.0040500, 0.0040447
3: 0.0022647, 0.0052722, 0.0022376, 0.0052781, -0.0030134, 0.0030346
4: 0.0016901, 0.0066729, 0.0016774, 0.0066890, -0.0049989, 0.0049956
5: 0.0025603, 0.0082570, 0.0025090, 0.0082681, -0.0057077, 0.0057480
6: -0.0040370, 0.0014595, -0.0040496, 0.0015070, -0.0055439, 0.0055091
7: -0.0088340, -0.0063972, -0.0088388, -0.0063753, -0.0024587, 0.0024415
8: 0.0036928, 0.0082358, 0.0036716, 0.0082387, -0.0045325, 0.0045512
9: -0.0050502, -0.0015705, -0.0050570, -0.0015392, -0.0035110, 0.0034864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030938, upper bound: 0.0029602
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031439, upper bound: 0.0030438
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0003246, 0.0020649, -0.0003490, 0.0020642, -0.0023888, 0.0024139
1: 0.9915064, 0.9965667, 0.9915080, 0.9966183, -0.0051119, 0.0050586
2: -0.0081682, -0.0041435, -0.0081804, -0.0041461, -0.0040222, 0.0040370
3: 0.0022820, 0.0052715, 0.0022515, 0.0052705, -0.0029885, 0.0030200
4: 0.0016918, 0.0066708, 0.0016938, 0.0066682, -0.0049764, 0.0049770
5: 0.0025932, 0.0082556, 0.0025354, 0.0082538, -0.0056606, 0.0057202
6: -0.0040353, 0.0014292, -0.0040333, 0.0014826, -0.0055179, 0.0054624
7: -0.0088334, -0.0064113, -0.0088326, -0.0063866, -0.0024468, 0.0024214
8: 0.0036956, 0.0082339, 0.0036990, 0.0082372, -0.0045251, 0.0045185
9: -0.0050493, -0.0015906, -0.0050482, -0.0015553, -0.0034940, 0.0034576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029174, upper bound: 0.0028688
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030549, upper bound: 0.0030265
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0004250, 0.0020652, -0.0024036, 0.0024906
1: 0.9915050, 0.9965960, 0.9915059, 0.9967793, -0.0052742, 0.0050901
2: -0.0081752, -0.0041413, -0.0082185, -0.0041426, -0.0040326, 0.0040772
3: 0.0022647, 0.0052722, 0.0021563, 0.0052718, -0.0030071, 0.0031159
4: 0.0016901, 0.0066729, 0.0016911, 0.0066717, -0.0049816, 0.0049818
5: 0.0025603, 0.0082570, 0.0023552, 0.0082561, -0.0056958, 0.0059019
6: -0.0040370, 0.0014595, -0.0040359, 0.0016492, -0.0056862, 0.0054955
7: -0.0088340, -0.0063972, -0.0088337, -0.0063095, -0.0025246, 0.0024364
8: 0.0036928, 0.0082358, 0.0036945, 0.0082473, -0.0045308, 0.0045253
9: -0.0050502, -0.0015705, -0.0050497, -0.0014452, -0.0036050, 0.0034791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030484, upper bound: 0.0028708
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031438, upper bound: 0.0030436
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002921, 0.0020715, -0.0002546, 0.0020637, -0.0023557, 0.0023261
1: 0.9914925, 0.9964977, 0.9915090, 0.9964183, -0.0049258, 0.0049887
2: -0.0081520, -0.0041208, -0.0081332, -0.0041478, -0.0040042, 0.0040124
3: 0.0023227, 0.0052797, 0.0023696, 0.0052699, -0.0029472, 0.0029101
4: 0.0016739, 0.0066934, 0.0016952, 0.0066664, -0.0049926, 0.0049982
5: 0.0026702, 0.0082711, 0.0027591, 0.0082526, -0.0055824, 0.0055120
6: -0.0040530, 0.0013579, -0.0040319, 0.0012757, -0.0053288, 0.0053898
7: -0.0088401, -0.0064442, -0.0088321, -0.0064823, -0.0023578, 0.0023879
8: 0.0036658, 0.0082296, 0.0037013, 0.0082246, -0.0045488, 0.0045156
9: -0.0050588, -0.0016376, -0.0050475, -0.0016920, -0.0033669, 0.0034099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0029610
time: 1.35 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030455, upper bound: 0.0030560
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003490, 0.0020642, -0.0002546, 0.0020637, -0.0024126, 0.0023187
1: 0.9915080, 0.9966183, 0.9915090, 0.9964183, -0.0049103, 0.0051093
2: -0.0081804, -0.0041461, -0.0081332, -0.0041478, -0.0040326, 0.0039872
3: 0.0022515, 0.0052705, 0.0023696, 0.0052699, -0.0030184, 0.0029009
4: 0.0016938, 0.0066682, 0.0016952, 0.0066664, -0.0049726, 0.0049730
5: 0.0025354, 0.0082538, 0.0027591, 0.0082526, -0.0057172, 0.0054947
6: -0.0040333, 0.0014826, -0.0040319, 0.0012757, -0.0053090, 0.0055145
7: -0.0088326, -0.0063866, -0.0088321, -0.0064823, -0.0023504, 0.0024455
8: 0.0036990, 0.0082372, 0.0037013, 0.0082246, -0.0045187, 0.0045204
9: -0.0050482, -0.0015553, -0.0050475, -0.0016920, -0.0033563, 0.0034922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0029610
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030455, upper bound: 0.0030560
time: 1.66 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003601, 0.0020702, -0.0002680, 0.0020643, -0.0024244, 0.0023382
1: 0.9914953, 0.9966418, 0.9915078, 0.9964467, -0.0049514, 0.0051340
2: -0.0081860, -0.0041252, -0.0081399, -0.0041456, -0.0040404, 0.0040147
3: 0.0022376, 0.0052781, 0.0023528, 0.0052707, -0.0030331, 0.0029252
4: 0.0016774, 0.0066890, 0.0016935, 0.0066686, -0.0049913, 0.0049955
5: 0.0025090, 0.0082681, 0.0027273, 0.0082541, -0.0057450, 0.0055407
6: -0.0040496, 0.0015070, -0.0040336, 0.0013051, -0.0053547, 0.0055405
7: -0.0088388, -0.0063753, -0.0088328, -0.0064687, -0.0023701, 0.0024575
8: 0.0036716, 0.0082387, 0.0036985, 0.0082264, -0.0045445, 0.0045216
9: -0.0050570, -0.0015392, -0.0050484, -0.0016726, -0.0033844, 0.0035092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030722
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031344
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004250, 0.0020652, -0.0002680, 0.0020643, -0.0024893, 0.0023331
1: 0.9915059, 0.9967793, 0.9915078, 0.9964467, -0.0049407, 0.0052715
2: -0.0082185, -0.0041426, -0.0081399, -0.0041456, -0.0040729, 0.0039973
3: 0.0021563, 0.0052718, 0.0023528, 0.0052707, -0.0031143, 0.0029189
4: 0.0016911, 0.0066717, 0.0016935, 0.0066686, -0.0049775, 0.0049782
5: 0.0023552, 0.0082561, 0.0027273, 0.0082541, -0.0058989, 0.0055288
6: -0.0040359, 0.0016492, -0.0040336, 0.0013051, -0.0053410, 0.0056828
7: -0.0088337, -0.0063095, -0.0088328, -0.0064687, -0.0023650, 0.0025233
8: 0.0036945, 0.0082473, 0.0036985, 0.0082264, -0.0045268, 0.0045268
9: -0.0050497, -0.0014452, -0.0050484, -0.0016726, -0.0033771, 0.0036032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030722
time: 1.30 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031344
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002921, 0.0020715, -0.0003246, 0.0020649, -0.0023570, 0.0023961
1: 0.9914925, 0.9964977, 0.9915064, 0.9965667, -0.0050742, 0.0049913
2: -0.0081520, -0.0041208, -0.0081682, -0.0041435, -0.0040085, 0.0040474
3: 0.0023227, 0.0052797, 0.0022820, 0.0052715, -0.0029488, 0.0029977
4: 0.0016739, 0.0066934, 0.0016918, 0.0066708, -0.0049969, 0.0050016
5: 0.0026702, 0.0082711, 0.0025932, 0.0082556, -0.0055853, 0.0056779
6: -0.0040530, 0.0013579, -0.0040353, 0.0014292, -0.0054822, 0.0053932
7: -0.0088401, -0.0064442, -0.0088334, -0.0064113, -0.0024288, 0.0023892
8: 0.0036658, 0.0082296, 0.0036956, 0.0082339, -0.0045554, 0.0045276
9: -0.0050588, -0.0016376, -0.0050493, -0.0015906, -0.0034682, 0.0034117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029259, upper bound: 0.0029441
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030265, upper bound: 0.0030575
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003601, 0.0020702, -0.0003385, 0.0020656, -0.0024257, 0.0024087
1: 0.9914953, 0.9966418, 0.9915050, 0.9965960, -0.0051007, 0.0051368
2: -0.0081860, -0.0041252, -0.0081752, -0.0041413, -0.0040447, 0.0040500
3: 0.0022376, 0.0052781, 0.0022647, 0.0052722, -0.0030346, 0.0030134
4: 0.0016774, 0.0066890, 0.0016901, 0.0066729, -0.0049956, 0.0049989
5: 0.0025090, 0.0082681, 0.0025603, 0.0082570, -0.0057480, 0.0057077
6: -0.0040496, 0.0015070, -0.0040370, 0.0014595, -0.0055091, 0.0055439
7: -0.0088388, -0.0063753, -0.0088340, -0.0063972, -0.0024415, 0.0024587
8: 0.0036716, 0.0082387, 0.0036928, 0.0082358, -0.0045512, 0.0045325
9: -0.0050570, -0.0015392, -0.0050502, -0.0015705, -0.0034864, 0.0035110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029259, upper bound: 0.0029441
time: 1.44 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030438, upper bound: 0.0031439
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003490, 0.0020642, -0.0003246, 0.0020649, -0.0024139, 0.0023888
1: 0.9915080, 0.9966183, 0.9915064, 0.9965667, -0.0050586, 0.0051119
2: -0.0081804, -0.0041461, -0.0081682, -0.0041435, -0.0040370, 0.0040222
3: 0.0022515, 0.0052705, 0.0022820, 0.0052715, -0.0030200, 0.0029885
4: 0.0016938, 0.0066682, 0.0016918, 0.0066708, -0.0049770, 0.0049764
5: 0.0025354, 0.0082538, 0.0025932, 0.0082556, -0.0057202, 0.0056606
6: -0.0040333, 0.0014826, -0.0040353, 0.0014292, -0.0054624, 0.0055179
7: -0.0088326, -0.0063866, -0.0088334, -0.0064113, -0.0024214, 0.0024468
8: 0.0036990, 0.0082372, 0.0036956, 0.0082339, -0.0045185, 0.0045251
9: -0.0050482, -0.0015553, -0.0050493, -0.0015906, -0.0034576, 0.0034940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028697, upper bound: 0.0029174
time: 1.14 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030317, upper bound: 0.0030545
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004250, 0.0020652, -0.0003385, 0.0020656, -0.0024906, 0.0024036
1: 0.9915059, 0.9967793, 0.9915050, 0.9965960, -0.0050901, 0.0052742
2: -0.0082185, -0.0041426, -0.0081752, -0.0041413, -0.0040772, 0.0040326
3: 0.0021563, 0.0052718, 0.0022647, 0.0052722, -0.0031159, 0.0030071
4: 0.0016911, 0.0066717, 0.0016901, 0.0066729, -0.0049818, 0.0049816
5: 0.0023552, 0.0082561, 0.0025603, 0.0082570, -0.0059019, 0.0056958
6: -0.0040359, 0.0016492, -0.0040370, 0.0014595, -0.0054955, 0.0056862
7: -0.0088337, -0.0063095, -0.0088340, -0.0063972, -0.0024364, 0.0025246
8: 0.0036945, 0.0082473, 0.0036928, 0.0082358, -0.0045253, 0.0045308
9: -0.0050497, -0.0014452, -0.0050502, -0.0015705, -0.0034791, 0.0036050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0030484
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030449, upper bound: 0.0031344
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002921, 0.0020715, -0.0003747, 0.0020725, -0.0023646, 0.0024462
1: 0.9914925, 0.9964977, 0.9914904, 0.9966727, -0.0051803, 0.0050073
2: -0.0081520, -0.0041208, -0.0081933, -0.0041173, -0.0040347, 0.0040725
3: 0.0023227, 0.0052797, 0.0022193, 0.0052809, -0.0029583, 0.0030604
4: 0.0016739, 0.0066934, 0.0016711, 0.0066969, -0.0050230, 0.0050223
5: 0.0026702, 0.0082711, 0.0024744, 0.0082735, -0.0056033, 0.0057967
6: -0.0040530, 0.0013579, -0.0040558, 0.0015390, -0.0055920, 0.0054137
7: -0.0088401, -0.0064442, -0.0088411, -0.0063605, -0.0024796, 0.0023968
8: 0.0036658, 0.0082296, 0.0036612, 0.0082406, -0.0045578, 0.0045559
9: -0.0050588, -0.0016376, -0.0050603, -0.0015181, -0.0035407, 0.0034227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030440, upper bound: 0.0029764
time: 1.34 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030849, upper bound: 0.0030816
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003601, 0.0020702, -0.0003878, 0.0020732, -0.0024333, 0.0024580
1: 0.9914953, 0.9966418, 0.9914889, 0.9967003, -0.0052050, 0.0051529
2: -0.0081860, -0.0041252, -0.0081998, -0.0041150, -0.0040710, 0.0040746
3: 0.0022376, 0.0052781, 0.0022030, 0.0052818, -0.0030442, 0.0030751
4: 0.0016774, 0.0066890, 0.0016693, 0.0066992, -0.0050218, 0.0050197
5: 0.0025090, 0.0082681, 0.0024435, 0.0082751, -0.0057660, 0.0058246
6: -0.0040496, 0.0015070, -0.0040576, 0.0015676, -0.0056171, 0.0055645
7: -0.0088388, -0.0063753, -0.0088418, -0.0063473, -0.0024915, 0.0024664
8: 0.0036716, 0.0082387, 0.0036583, 0.0082424, -0.0045541, 0.0045613
9: -0.0050570, -0.0015392, -0.0050612, -0.0014992, -0.0035578, 0.0035220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030661, upper bound: 0.0031350
time: 1.46 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030910, upper bound: 0.0031545
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003490, 0.0020642, -0.0003747, 0.0020725, -0.0024215, 0.0024389
1: 0.9915080, 0.9966183, 0.9914904, 0.9966727, -0.0051647, 0.0051278
2: -0.0081804, -0.0041461, -0.0081933, -0.0041173, -0.0040632, 0.0040473
3: 0.0022515, 0.0052705, 0.0022193, 0.0052809, -0.0030294, 0.0030512
4: 0.0016938, 0.0066682, 0.0016711, 0.0066969, -0.0050031, 0.0049971
5: 0.0025354, 0.0082538, 0.0024744, 0.0082735, -0.0057381, 0.0057794
6: -0.0040333, 0.0014826, -0.0040558, 0.0015390, -0.0055722, 0.0055384
7: -0.0088326, -0.0063866, -0.0088411, -0.0063605, -0.0024722, 0.0024545
8: 0.0036990, 0.0082372, 0.0036612, 0.0082406, -0.0045274, 0.0045608
9: -0.0050482, -0.0015553, -0.0050603, -0.0015181, -0.0035302, 0.0035050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029983, upper bound: 0.0028325
time: 1.11 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031023, upper bound: 0.0030770
time: 1.64 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004250, 0.0020652, -0.0003878, 0.0020732, -0.0024982, 0.0024529
1: 0.9915059, 0.9967793, 0.9914889, 0.9967003, -0.0051944, 0.0052903
2: -0.0082185, -0.0041426, -0.0081998, -0.0041150, -0.0041035, 0.0040572
3: 0.0021563, 0.0052718, 0.0022030, 0.0052818, -0.0031254, 0.0030688
4: 0.0016911, 0.0066717, 0.0016693, 0.0066992, -0.0050081, 0.0050024
5: 0.0023552, 0.0082561, 0.0024435, 0.0082751, -0.0059199, 0.0058126
6: -0.0040359, 0.0016492, -0.0040576, 0.0015676, -0.0056035, 0.0057068
7: -0.0088337, -0.0063095, -0.0088418, -0.0063473, -0.0024864, 0.0025323
8: 0.0036945, 0.0082473, 0.0036583, 0.0082424, -0.0045367, 0.0045666
9: -0.0050497, -0.0014452, -0.0050612, -0.0014992, -0.0035505, 0.0036160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030226, upper bound: 0.0030507
time: 1.27 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031075, upper bound: 0.0031484
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003747, 0.0020725, -0.0003490, 0.0020642, -0.0024389, 0.0024215
1: 0.9914904, 0.9966727, 0.9915080, 0.9966183, -0.0051278, 0.0051647
2: -0.0081933, -0.0041173, -0.0081804, -0.0041461, -0.0040473, 0.0040632
3: 0.0022193, 0.0052809, 0.0022515, 0.0052705, -0.0030512, 0.0030294
4: 0.0016711, 0.0066969, 0.0016938, 0.0066682, -0.0049971, 0.0050031
5: 0.0024744, 0.0082735, 0.0025354, 0.0082538, -0.0057794, 0.0057381
6: -0.0040558, 0.0015390, -0.0040333, 0.0014826, -0.0055384, 0.0055722
7: -0.0088411, -0.0063605, -0.0088326, -0.0063866, -0.0024545, 0.0024722
8: 0.0036612, 0.0082406, 0.0036990, 0.0082372, -0.0045608, 0.0045274
9: -0.0050603, -0.0015181, -0.0050482, -0.0015553, -0.0035050, 0.0035302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0030445
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030287, upper bound: 0.0031549
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003878, 0.0020732, -0.0004250, 0.0020652, -0.0024529, 0.0024982
1: 0.9914889, 0.9967003, 0.9915059, 0.9967793, -0.0052903, 0.0051944
2: -0.0081998, -0.0041150, -0.0082185, -0.0041426, -0.0040572, 0.0041035
3: 0.0022030, 0.0052818, 0.0021563, 0.0052718, -0.0030688, 0.0031254
4: 0.0016693, 0.0066992, 0.0016911, 0.0066717, -0.0050024, 0.0050081
5: 0.0024435, 0.0082751, 0.0023552, 0.0082561, -0.0058126, 0.0059199
6: -0.0040576, 0.0015676, -0.0040359, 0.0016492, -0.0057068, 0.0056035
7: -0.0088418, -0.0063473, -0.0088337, -0.0063095, -0.0025323, 0.0024864
8: 0.0036583, 0.0082424, 0.0036945, 0.0082473, -0.0045666, 0.0045367
9: -0.0050612, -0.0014992, -0.0050497, -0.0014452, -0.0036160, 0.0035505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030160, upper bound: 0.0030739
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030829, upper bound: 0.0031616
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003490, 0.0020642, -0.0004362, 0.0020677, -0.0024167, 0.0025004
1: 0.9915080, 0.9966183, 0.9915005, 0.9968030, -0.0052950, 0.0051178
2: -0.0081804, -0.0041461, -0.0082241, -0.0041338, -0.0040467, 0.0040780
3: 0.0022515, 0.0052705, 0.0021424, 0.0052750, -0.0030235, 0.0031281
4: 0.0016938, 0.0066682, 0.0016841, 0.0066805, -0.0049867, 0.0049841
5: 0.0025354, 0.0082538, 0.0023287, 0.0082622, -0.0057268, 0.0059251
6: -0.0040333, 0.0014826, -0.0040429, 0.0016737, -0.0057070, 0.0055255
7: -0.0088326, -0.0063866, -0.0088363, -0.0062982, -0.0025345, 0.0024497
8: 0.0036990, 0.0082372, 0.0036829, 0.0082488, -0.0045254, 0.0045356
9: -0.0050482, -0.0015553, -0.0050534, -0.0014291, -0.0036192, 0.0034981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029848, upper bound: 0.0028325
time: 1.22 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031023, upper bound: 0.0030764
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004250, 0.0020652, -0.0004500, 0.0020684, -0.0024934, 0.0025152
1: 0.9915059, 0.9967793, 0.9914991, 0.9968323, -0.0053263, 0.0052802
2: -0.0082185, -0.0041426, -0.0082310, -0.0041315, -0.0040870, 0.0040884
3: 0.0021563, 0.0052718, 0.0021251, 0.0052758, -0.0031194, 0.0031467
4: 0.0016911, 0.0066717, 0.0016823, 0.0066827, -0.0049916, 0.0049893
5: 0.0023552, 0.0082561, 0.0022959, 0.0082637, -0.0059086, 0.0059602
6: -0.0040359, 0.0016492, -0.0040447, 0.0017040, -0.0057400, 0.0056939
7: -0.0088337, -0.0063095, -0.0088369, -0.0062842, -0.0025495, 0.0025274
8: 0.0036945, 0.0082473, 0.0036799, 0.0082507, -0.0045329, 0.0045414
9: -0.0050497, -0.0014452, -0.0050543, -0.0014091, -0.0036406, 0.0036091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030110, upper bound: 0.0030506
time: 1.28 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031075, upper bound: 0.0031484
time: 1.15 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.72 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029895, upper bound: 0.0028433
IS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030506, upper bound: 0.0029421
IS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030268, upper bound: 0.0030382
IS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030710, upper bound: 0.0030710
IS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0029211
IS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0030468
IS_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0029593
IS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030440, upper bound: 0.0030626
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029211, upper bound: 0.0026852
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030468, upper bound: 0.0029169
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029593, upper bound: 0.0028708
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030626, upper bound: 0.0030441
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029047, upper bound: 0.0026852
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030317, upper bound: 0.0029169
IS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029410, upper bound: 0.0028708
IS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030449, upper bound: 0.0030437
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029610, upper bound: 0.0029916
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0030455
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029610, upper bound: 0.0029916
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0030460
IS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030722, upper bound: 0.0030166
IS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031344, upper bound: 0.0030595
IS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030722, upper bound: 0.0030166
IS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031344, upper bound: 0.0030605
IS_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029441, upper bound: 0.0029259
IS_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030575, upper bound: 0.0030265
IS_A1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030938, upper bound: 0.0029602
IS_A1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031439, upper bound: 0.0030438
IS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029174, upper bound: 0.0028688
IS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030549, upper bound: 0.0030265
IS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030484, upper bound: 0.0028708
IS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031438, upper bound: 0.0030436
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0029610
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030455, upper bound: 0.0030560
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0029610
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030455, upper bound: 0.0030560
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030722
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031344
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030722
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031344
IS_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029259, upper bound: 0.0029441
IS_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030265, upper bound: 0.0030575
IS_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029259, upper bound: 0.0029441
IS_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030438, upper bound: 0.0031439
IS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0028697, upper bound: 0.0029174
IS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030317, upper bound: 0.0030545
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0030484
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030449, upper bound: 0.0031344
IS_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030440, upper bound: 0.0029764
IS_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030849, upper bound: 0.0030816
IS_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030661, upper bound: 0.0031350
IS_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030910, upper bound: 0.0031545
IS_A2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029983, upper bound: 0.0028325
IS_A2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031023, upper bound: 0.0030770
IS_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030226, upper bound: 0.0030507
IS_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031075, upper bound: 0.0031484
IS_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0030445
IS_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030287, upper bound: 0.0031549
IS_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030160, upper bound: 0.0030739
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030829, upper bound: 0.0031616
IS_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0029848, upper bound: 0.0028325
IS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031023, upper bound: 0.0030764
IS_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0030110, upper bound: 0.0030506
IS_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 1, lower bound: -0.0031075, upper bound: 0.0031484

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002327, 0.0019566, -0.0002533, 0.0020477, -0.0022804, 0.0022099
1: 0.9917359, 0.9963719, 0.9915429, 0.9964155, -0.0046796, 0.0048290
2: -0.0081223, -0.0045176, -0.0081326, -0.0042030, -0.0039193, 0.0036149
3: 0.0023970, 0.0051359, 0.0023712, 0.0052499, -0.0028529, 0.0027647
4: 0.0019875, 0.0062975, 0.0017388, 0.0066114, -0.0046239, 0.0045587
5: 0.0028110, 0.0079988, 0.0027621, 0.0082147, -0.0054037, 0.0052367
6: -0.0037419, 0.0012277, -0.0039886, 0.0012729, -0.0050148, 0.0052164
7: -0.0087236, -0.0065045, -0.0088159, -0.0064836, -0.0022400, 0.0023115
8: 0.0041872, 0.0082217, 0.0037738, 0.0082244, -0.0040219, 0.0044162
9: -0.0048925, -0.0017237, -0.0050244, -0.0016938, -0.0031987, 0.0033007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028976, upper bound: 0.0027993
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028691, upper bound: 0.0027149
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0001676, 0.0020273, -0.0002546, 0.0020637, -0.0022313, 0.0022819
1: 0.9915861, 0.9962342, 0.9915090, 0.9964183, -0.0048322, 0.0047252
2: -0.0080897, -0.0042735, -0.0081332, -0.0041478, -0.0039419, 0.0038598
3: 0.0024784, 0.0052244, 0.0023696, 0.0052699, -0.0027915, 0.0028548
4: 0.0017945, 0.0065411, 0.0016952, 0.0066664, -0.0048719, 0.0048459
5: 0.0029651, 0.0081664, 0.0027591, 0.0082526, -0.0052875, 0.0054073
6: -0.0039333, 0.0010853, -0.0040319, 0.0012757, -0.0052091, 0.0051171
7: -0.0087953, -0.0065704, -0.0088321, -0.0064823, -0.0023130, 0.0022617
8: 0.0038664, 0.0082130, 0.0037013, 0.0082246, -0.0043412, 0.0044980
9: -0.0049948, -0.0018178, -0.0050475, -0.0016920, -0.0033029, 0.0032297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030101, upper bound: 0.0028998
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030101, upper bound: 0.0029421
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003042, 0.0019554, -0.0002667, 0.0020483, -0.0023525, 0.0022221
1: 0.9917383, 0.9965233, 0.9915415, 0.9964439, -0.0047056, 0.0049818
2: -0.0081581, -0.0045218, -0.0081393, -0.0042008, -0.0039573, 0.0036175
3: 0.0023075, 0.0051344, 0.0023545, 0.0052507, -0.0029432, 0.0027800
4: 0.0019908, 0.0062933, 0.0017371, 0.0066136, -0.0046228, 0.0045563
5: 0.0026415, 0.0079960, 0.0027304, 0.0082162, -0.0055747, 0.0052656
6: -0.0037386, 0.0013845, -0.0039903, 0.0013023, -0.0050409, 0.0053748
7: -0.0087224, -0.0064320, -0.0088166, -0.0064700, -0.0022524, 0.0023846
8: 0.0041927, 0.0082312, 0.0037709, 0.0082262, -0.0040189, 0.0044219
9: -0.0048908, -0.0016201, -0.0050253, -0.0016744, -0.0032163, 0.0034052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029298, upper bound: 0.0029546
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029102, upper bound: 0.0029012
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002399, 0.0020290, -0.0002680, 0.0020643, -0.0023042, 0.0022969
1: 0.9915825, 0.9963874, 0.9915078, 0.9964467, -0.0048642, 0.0048796
2: -0.0081259, -0.0042677, -0.0081399, -0.0041456, -0.0039803, 0.0038722
3: 0.0023879, 0.0052264, 0.0023528, 0.0052707, -0.0028828, 0.0028736
4: 0.0017900, 0.0065468, 0.0016935, 0.0066686, -0.0048786, 0.0048533
5: 0.0027937, 0.0081703, 0.0027273, 0.0082541, -0.0054603, 0.0054429
6: -0.0039378, 0.0012437, -0.0040336, 0.0013051, -0.0052429, 0.0052773
7: -0.0087969, -0.0064971, -0.0088328, -0.0064687, -0.0023282, 0.0023357
8: 0.0038589, 0.0082226, 0.0036985, 0.0082264, -0.0043500, 0.0045041
9: -0.0049972, -0.0017131, -0.0050484, -0.0016726, -0.0033247, 0.0033353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030382, upper bound: 0.0030268
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030382, upper bound: 0.0030710
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002533, 0.0020477, -0.0002836, 0.0019403, -0.0021936, 0.0023313
1: 0.9915429, 0.9964155, 0.9917703, 0.9964798, -0.0049369, 0.0046452
2: -0.0081326, -0.0042030, -0.0081478, -0.0045740, -0.0035586, 0.0039448
3: 0.0023712, 0.0052499, 0.0023332, 0.0051155, -0.0027443, 0.0029167
4: 0.0017388, 0.0066114, 0.0020320, 0.0062413, -0.0045025, 0.0045794
5: 0.0027621, 0.0082147, 0.0026902, 0.0079602, -0.0051981, 0.0055245
6: -0.0039886, 0.0012729, -0.0036977, 0.0013394, -0.0053280, 0.0049706
7: -0.0088159, -0.0064836, -0.0087071, -0.0064528, -0.0023631, 0.0022235
8: 0.0037738, 0.0082244, 0.0042612, 0.0082285, -0.0044195, 0.0039513
9: -0.0050244, -0.0016938, -0.0048689, -0.0016499, -0.0033745, 0.0031751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026397, upper bound: 0.0028039
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023703, upper bound: 0.0026745
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002546, 0.0020637, -0.0002344, 0.0020296, -0.0022842, 0.0022981
1: 0.9915090, 0.9964183, 0.9915812, 0.9963756, -0.0048665, 0.0048371
2: -0.0081332, -0.0041478, -0.0081231, -0.0042655, -0.0038678, 0.0039753
3: 0.0023696, 0.0052699, 0.0023948, 0.0052273, -0.0028577, 0.0028750
4: 0.0016952, 0.0066664, 0.0017882, 0.0065491, -0.0048539, 0.0048782
5: 0.0027591, 0.0082526, 0.0028069, 0.0081718, -0.0054127, 0.0054457
6: -0.0040319, 0.0012757, -0.0039396, 0.0012315, -0.0052634, 0.0052153
7: -0.0088321, -0.0064823, -0.0087976, -0.0065027, -0.0023294, 0.0023153
8: 0.0037013, 0.0082246, 0.0038559, 0.0082219, -0.0045048, 0.0043555
9: -0.0050475, -0.0016920, -0.0049982, -0.0017212, -0.0033263, 0.0033062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028536, upper bound: 0.0029916
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028536, upper bound: 0.0030468
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002667, 0.0020483, -0.0003602, 0.0019436, -0.0022102, 0.0024085
1: 0.9915415, 0.9964439, 0.9917633, 0.9966419, -0.0051004, 0.0046806
2: -0.0081393, -0.0042008, -0.0081861, -0.0045626, -0.0035766, 0.0039853
3: 0.0023545, 0.0052507, 0.0022374, 0.0051196, -0.0027652, 0.0030133
4: 0.0017371, 0.0066136, 0.0019143, 0.0062526, -0.0045155, 0.0046993
5: 0.0027304, 0.0082162, 0.0025088, 0.0079680, -0.0052376, 0.0057074
6: -0.0039903, 0.0013023, -0.0037066, 0.0015072, -0.0054975, 0.0050089
7: -0.0088166, -0.0064700, -0.0087104, -0.0063752, -0.0024414, 0.0022404
8: 0.0037709, 0.0082262, 0.0042464, 0.0082387, -0.0044260, 0.0039688
9: -0.0050253, -0.0016744, -0.0048737, -0.0015391, -0.0034862, 0.0031992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028089, upper bound: 0.0028548
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026595, upper bound: 0.0028023
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002680, 0.0020643, -0.0003130, 0.0020349, -0.0023028, 0.0023773
1: 0.9915078, 0.9964467, 0.9915700, 0.9965421, -0.0050343, 0.0048767
2: -0.0081399, -0.0041456, -0.0081625, -0.0042473, -0.0038926, 0.0040168
3: 0.0023528, 0.0052707, 0.0022965, 0.0052338, -0.0028810, 0.0029742
4: 0.0016935, 0.0066686, 0.0017739, 0.0065672, -0.0048737, 0.0048948
5: 0.0027273, 0.0082541, 0.0026206, 0.0081843, -0.0054569, 0.0056335
6: -0.0040336, 0.0013051, -0.0039538, 0.0014038, -0.0054374, 0.0052589
7: -0.0088328, -0.0064687, -0.0088029, -0.0064230, -0.0024097, 0.0023342
8: 0.0036985, 0.0082264, 0.0038321, 0.0082324, -0.0045116, 0.0043819
9: -0.0050484, -0.0016726, -0.0050058, -0.0016073, -0.0034411, 0.0033332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029608, upper bound: 0.0030167
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029608, upper bound: 0.0030626
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002836, 0.0019403, -0.0002533, 0.0020477, -0.0023313, 0.0021936
1: 0.9917703, 0.9964798, 0.9915429, 0.9964155, -0.0046452, 0.0049369
2: -0.0081478, -0.0045740, -0.0081326, -0.0042030, -0.0039448, 0.0035586
3: 0.0023332, 0.0051155, 0.0023712, 0.0052499, -0.0029167, 0.0027443
4: 0.0020320, 0.0062413, 0.0017388, 0.0066114, -0.0045794, 0.0045025
5: 0.0026902, 0.0079602, 0.0027621, 0.0082147, -0.0055245, 0.0051981
6: -0.0036977, 0.0013394, -0.0039886, 0.0012729, -0.0049706, 0.0053280
7: -0.0087071, -0.0064528, -0.0088159, -0.0064836, -0.0022235, 0.0023631
8: 0.0042612, 0.0082285, 0.0037738, 0.0082244, -0.0039513, 0.0044195
9: -0.0048689, -0.0016499, -0.0050244, -0.0016938, -0.0031751, 0.0033745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028039, upper bound: 0.0026397
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026745, upper bound: 0.0023703
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002344, 0.0020296, -0.0002546, 0.0020637, -0.0022981, 0.0022842
1: 0.9915812, 0.9963756, 0.9915090, 0.9964183, -0.0048371, 0.0048665
2: -0.0081231, -0.0042655, -0.0081332, -0.0041478, -0.0039753, 0.0038678
3: 0.0023948, 0.0052273, 0.0023696, 0.0052699, -0.0028750, 0.0028577
4: 0.0017882, 0.0065491, 0.0016952, 0.0066664, -0.0048782, 0.0048539
5: 0.0028069, 0.0081718, 0.0027591, 0.0082526, -0.0054457, 0.0054127
6: -0.0039396, 0.0012315, -0.0040319, 0.0012757, -0.0052153, 0.0052634
7: -0.0087976, -0.0065027, -0.0088321, -0.0064823, -0.0023153, 0.0023294
8: 0.0038559, 0.0082219, 0.0037013, 0.0082246, -0.0043555, 0.0045048
9: -0.0049982, -0.0017212, -0.0050475, -0.0016920, -0.0033062, 0.0033263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0028536
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0029169
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003602, 0.0019436, -0.0002667, 0.0020483, -0.0024085, 0.0022102
1: 0.9917633, 0.9966419, 0.9915415, 0.9964439, -0.0046806, 0.0051004
2: -0.0081861, -0.0045626, -0.0081393, -0.0042008, -0.0039853, 0.0035766
3: 0.0022374, 0.0051196, 0.0023545, 0.0052507, -0.0030133, 0.0027652
4: 0.0019143, 0.0062526, 0.0017371, 0.0066136, -0.0046993, 0.0045155
5: 0.0025088, 0.0079680, 0.0027304, 0.0082162, -0.0057074, 0.0052376
6: -0.0037066, 0.0015072, -0.0039903, 0.0013023, -0.0050089, 0.0054975
7: -0.0087104, -0.0063752, -0.0088166, -0.0064700, -0.0022404, 0.0024414
8: 0.0042464, 0.0082387, 0.0037709, 0.0082262, -0.0039688, 0.0044260
9: -0.0048737, -0.0015391, -0.0050253, -0.0016744, -0.0031992, 0.0034862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028548, upper bound: 0.0028088
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028022, upper bound: 0.0026595
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0003130, 0.0020349, -0.0002680, 0.0020643, -0.0023773, 0.0023028
1: 0.9915700, 0.9965421, 0.9915078, 0.9964467, -0.0048767, 0.0050343
2: -0.0081625, -0.0042473, -0.0081399, -0.0041456, -0.0040168, 0.0038926
3: 0.0022965, 0.0052338, 0.0023528, 0.0052707, -0.0029742, 0.0028810
4: 0.0017739, 0.0065672, 0.0016935, 0.0066686, -0.0048948, 0.0048737
5: 0.0026206, 0.0081843, 0.0027273, 0.0082541, -0.0056335, 0.0054569
6: -0.0039538, 0.0014038, -0.0040336, 0.0013051, -0.0052589, 0.0054374
7: -0.0088029, -0.0064230, -0.0088328, -0.0064687, -0.0023342, 0.0024097
8: 0.0038321, 0.0082324, 0.0036985, 0.0082264, -0.0043819, 0.0045116
9: -0.0050058, -0.0016073, -0.0050484, -0.0016726, -0.0033332, 0.0034411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030167, upper bound: 0.0029608
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030167, upper bound: 0.0030441
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002836, 0.0019403, -0.0003234, 0.0020472, -0.0023308, 0.0022637
1: 0.9917703, 0.9964798, 0.9915439, 0.9965640, -0.0047937, 0.0049359
2: -0.0081478, -0.0045740, -0.0081676, -0.0042046, -0.0039431, 0.0035937
3: 0.0023332, 0.0051155, 0.0022835, 0.0052493, -0.0029161, 0.0028320
4: 0.0020320, 0.0062413, 0.0017401, 0.0066098, -0.0045777, 0.0045012
5: 0.0026902, 0.0079602, 0.0025960, 0.0082136, -0.0055233, 0.0053642
6: -0.0036977, 0.0013394, -0.0039873, 0.0014265, -0.0051242, 0.0053267
7: -0.0087071, -0.0064528, -0.0088155, -0.0064125, -0.0022945, 0.0023626
8: 0.0042612, 0.0082285, 0.0037760, 0.0082338, -0.0039515, 0.0044166
9: -0.0048689, -0.0016499, -0.0050237, -0.0015924, -0.0032765, 0.0033738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027812, upper bound: 0.0026396
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026543, upper bound: 0.0023703
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002344, 0.0020296, -0.0003246, 0.0020649, -0.0022993, 0.0023542
1: 0.9915812, 0.9963756, 0.9915064, 0.9965667, -0.0049855, 0.0048692
2: -0.0081231, -0.0042655, -0.0081682, -0.0041435, -0.0039797, 0.0039028
3: 0.0023948, 0.0052273, 0.0022820, 0.0052715, -0.0028766, 0.0029453
4: 0.0017882, 0.0065491, 0.0016918, 0.0066708, -0.0048826, 0.0048573
5: 0.0028069, 0.0081718, 0.0025932, 0.0082556, -0.0054487, 0.0055787
6: -0.0039396, 0.0012315, -0.0040353, 0.0014292, -0.0053688, 0.0052668
7: -0.0087976, -0.0065027, -0.0088334, -0.0064113, -0.0023863, 0.0023307
8: 0.0038559, 0.0082219, 0.0036956, 0.0082339, -0.0043572, 0.0045081
9: -0.0049982, -0.0017212, -0.0050493, -0.0015906, -0.0034076, 0.0033282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028696, upper bound: 0.0028240
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028696, upper bound: 0.0028240
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003602, 0.0019436, -0.0003372, 0.0020478, -0.0024080, 0.0022808
1: 0.9917633, 0.9966419, 0.9915425, 0.9965934, -0.0048301, 0.0050994
2: -0.0081861, -0.0045626, -0.0081746, -0.0042025, -0.0039836, 0.0036119
3: 0.0022374, 0.0051196, 0.0022662, 0.0052501, -0.0030126, 0.0028534
4: 0.0019143, 0.0062526, 0.0017384, 0.0066119, -0.0046976, 0.0045142
5: 0.0025088, 0.0079680, 0.0025633, 0.0082150, -0.0057063, 0.0054047
6: -0.0037066, 0.0015072, -0.0039890, 0.0014568, -0.0051634, 0.0054962
7: -0.0087104, -0.0063752, -0.0088161, -0.0063985, -0.0023119, 0.0024409
8: 0.0042464, 0.0082387, 0.0037732, 0.0082356, -0.0039687, 0.0044228
9: -0.0048737, -0.0015391, -0.0050246, -0.0015724, -0.0033013, 0.0034855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028262, upper bound: 0.0028087
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027764, upper bound: 0.0026595
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0003130, 0.0020349, -0.0003385, 0.0020656, -0.0023786, 0.0023733
1: 0.9915700, 0.9965421, 0.9915050, 0.9965960, -0.0050260, 0.0050371
2: -0.0081625, -0.0042473, -0.0081752, -0.0041413, -0.0040212, 0.0039279
3: 0.0022965, 0.0052338, 0.0022647, 0.0052722, -0.0029758, 0.0029692
4: 0.0017739, 0.0065672, 0.0016901, 0.0066729, -0.0048991, 0.0048771
5: 0.0026206, 0.0081843, 0.0025603, 0.0082570, -0.0056365, 0.0056240
6: -0.0039538, 0.0014038, -0.0040370, 0.0014595, -0.0054134, 0.0054408
7: -0.0088029, -0.0064230, -0.0088340, -0.0063972, -0.0024057, 0.0024110
8: 0.0038321, 0.0082324, 0.0036928, 0.0082358, -0.0043829, 0.0045142
9: -0.0050058, -0.0016073, -0.0050502, -0.0015705, -0.0034352, 0.0034429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0029410
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0030437
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003183, 0.0019577, -0.0002908, 0.0020564, -0.0023747, 0.0022485
1: 0.9917334, 0.9965532, 0.9915244, 0.9964951, -0.0047617, 0.0050288
2: -0.0081651, -0.0045138, -0.0081514, -0.0041727, -0.0039924, 0.0036375
3: 0.0022898, 0.0051373, 0.0023243, 0.0052608, -0.0029710, 0.0028130
4: 0.0019826, 0.0063013, 0.0017149, 0.0066416, -0.0046589, 0.0045864
5: 0.0026081, 0.0080015, 0.0026732, 0.0082354, -0.0056274, 0.0053283
6: -0.0037449, 0.0014154, -0.0040123, 0.0013552, -0.0051001, 0.0054277
7: -0.0087247, -0.0064177, -0.0088248, -0.0064455, -0.0022792, 0.0024071
8: 0.0041822, 0.0082331, 0.0037341, 0.0082294, -0.0040305, 0.0044634
9: -0.0048941, -0.0015997, -0.0050370, -0.0016395, -0.0032546, 0.0034373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029515, upper bound: 0.0029315
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028995, upper bound: 0.0028912
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002531, 0.0020312, -0.0002921, 0.0020715, -0.0023246, 0.0023233
1: 0.9915779, 0.9964153, 0.9914925, 0.9964977, -0.0049198, 0.0049229
2: -0.0081325, -0.0042600, -0.0081520, -0.0041208, -0.0040117, 0.0038920
3: 0.0023714, 0.0052293, 0.0023227, 0.0052797, -0.0029083, 0.0029066
4: 0.0017839, 0.0065546, 0.0016739, 0.0066934, -0.0049095, 0.0048807
5: 0.0027625, 0.0081756, 0.0026702, 0.0082711, -0.0055086, 0.0055054
6: -0.0039439, 0.0012726, -0.0040530, 0.0013579, -0.0053019, 0.0053256
7: -0.0087992, -0.0064838, -0.0088401, -0.0064442, -0.0023550, 0.0023563
8: 0.0038487, 0.0082244, 0.0036658, 0.0082296, -0.0043629, 0.0045414
9: -0.0050005, -0.0016941, -0.0050588, -0.0016376, -0.0033629, 0.0033647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029722, upper bound: 0.0029882
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029722, upper bound: 0.0030491
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003183, 0.0019577, -0.0003477, 0.0020480, -0.0023663, 0.0023054
1: 0.9917334, 0.9965532, 0.9915422, 0.9966156, -0.0048822, 0.0050110
2: -0.0081651, -0.0045138, -0.0081798, -0.0042020, -0.0039631, 0.0036660
3: 0.0022898, 0.0051373, 0.0022531, 0.0052503, -0.0029604, 0.0028842
4: 0.0019826, 0.0063013, 0.0017380, 0.0066124, -0.0046298, 0.0045633
5: 0.0026081, 0.0080015, 0.0025384, 0.0082154, -0.0056073, 0.0054630
6: -0.0037449, 0.0014154, -0.0039894, 0.0014798, -0.0052247, 0.0054048
7: -0.0087247, -0.0064177, -0.0088162, -0.0063879, -0.0023368, 0.0023985
8: 0.0041822, 0.0082331, 0.0037725, 0.0082370, -0.0040353, 0.0044276
9: -0.0048941, -0.0015997, -0.0050248, -0.0015572, -0.0033370, 0.0034251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028992, upper bound: 0.0029065
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028481, upper bound: 0.0028715
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002531, 0.0020312, -0.0003490, 0.0020642, -0.0023173, 0.0023802
1: 0.9915779, 0.9964153, 0.9915080, 0.9966183, -0.0050404, 0.0049073
2: -0.0081325, -0.0042600, -0.0081804, -0.0041461, -0.0039865, 0.0039205
3: 0.0023714, 0.0052293, 0.0022515, 0.0052705, -0.0028991, 0.0029778
4: 0.0017839, 0.0065546, 0.0016938, 0.0066682, -0.0048844, 0.0048607
5: 0.0027625, 0.0081756, 0.0025354, 0.0082538, -0.0054913, 0.0056402
6: -0.0039439, 0.0012726, -0.0040333, 0.0014826, -0.0054265, 0.0053058
7: -0.0087992, -0.0064838, -0.0088326, -0.0063866, -0.0024126, 0.0023489
8: 0.0038487, 0.0082244, 0.0036990, 0.0082372, -0.0043653, 0.0045114
9: -0.0050005, -0.0016941, -0.0050482, -0.0015553, -0.0034452, 0.0033542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0029211
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0030460
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003316, 0.0019584, -0.0003588, 0.0020549, -0.0023865, 0.0023172
1: 0.9917320, 0.9965814, 0.9915276, 0.9966391, -0.0049071, 0.0050538
2: -0.0081717, -0.0045115, -0.0081854, -0.0041781, -0.0039937, 0.0036739
3: 0.0022733, 0.0051382, 0.0022392, 0.0052589, -0.0029857, 0.0028990
4: 0.0019610, 0.0063036, 0.0017191, 0.0066363, -0.0046753, 0.0045845
5: 0.0025766, 0.0080031, 0.0025120, 0.0082318, -0.0056552, 0.0054910
6: -0.0037467, 0.0014445, -0.0040081, 0.0015042, -0.0052509, 0.0054526
7: -0.0087254, -0.0064042, -0.0088232, -0.0063766, -0.0023488, 0.0024190
8: 0.0041792, 0.0082349, 0.0037411, 0.0082385, -0.0040367, 0.0044583
9: -0.0048951, -0.0015805, -0.0050348, -0.0015411, -0.0033540, 0.0034543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029981, upper bound: 0.0028433
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029981, upper bound: 0.0028433
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002665, 0.0020318, -0.0003601, 0.0020702, -0.0023367, 0.0023919
1: 0.9915764, 0.9964435, 0.9914953, 0.9966418, -0.0050654, 0.0049483
2: -0.0081392, -0.0042578, -0.0081860, -0.0041252, -0.0040140, 0.0039282
3: 0.0023547, 0.0052301, 0.0022376, 0.0052781, -0.0029234, 0.0029924
4: 0.0017821, 0.0065567, 0.0016774, 0.0066890, -0.0049069, 0.0048794
5: 0.0027308, 0.0081771, 0.0025090, 0.0082681, -0.0055373, 0.0056681
6: -0.0039456, 0.0013019, -0.0040496, 0.0015070, -0.0054526, 0.0053515
7: -0.0087999, -0.0064702, -0.0088388, -0.0063753, -0.0024245, 0.0023686
8: 0.0038458, 0.0082262, 0.0036716, 0.0082387, -0.0043687, 0.0045372
9: -0.0050014, -0.0016747, -0.0050570, -0.0015392, -0.0034622, 0.0033823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030628, upper bound: 0.0029421
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030628, upper bound: 0.0030662
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003316, 0.0019584, -0.0004238, 0.0020491, -0.0023806, 0.0023822
1: 0.9917320, 0.9965814, 0.9915399, 0.9967766, -0.0050446, 0.0050415
2: -0.0081717, -0.0045115, -0.0082179, -0.0041982, -0.0039735, 0.0037064
3: 0.0022733, 0.0051382, 0.0021579, 0.0052516, -0.0029784, 0.0029802
4: 0.0019610, 0.0063036, 0.0017351, 0.0066162, -0.0046552, 0.0045685
5: 0.0025766, 0.0080031, 0.0023581, 0.0082180, -0.0056414, 0.0056449
6: -0.0037467, 0.0014445, -0.0039924, 0.0016465, -0.0053932, 0.0054368
7: -0.0087254, -0.0064042, -0.0088173, -0.0063108, -0.0024147, 0.0024131
8: 0.0041792, 0.0082349, 0.0037675, 0.0082472, -0.0040420, 0.0044350
9: -0.0048951, -0.0015805, -0.0050264, -0.0014470, -0.0034481, 0.0034459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0028431
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0030166
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002665, 0.0020318, -0.0004250, 0.0020652, -0.0023317, 0.0024569
1: 0.9915764, 0.9964435, 0.9915059, 0.9967793, -0.0052028, 0.0049376
2: -0.0081392, -0.0042578, -0.0082185, -0.0041426, -0.0039966, 0.0039607
3: 0.0023547, 0.0052301, 0.0021563, 0.0052718, -0.0029171, 0.0030737
4: 0.0017821, 0.0065567, 0.0016911, 0.0066717, -0.0048895, 0.0048656
5: 0.0027308, 0.0081771, 0.0023552, 0.0082561, -0.0055254, 0.0058220
6: -0.0039456, 0.0013019, -0.0040359, 0.0016492, -0.0055949, 0.0053379
7: -0.0087999, -0.0064702, -0.0088337, -0.0063095, -0.0024904, 0.0023635
8: 0.0038458, 0.0082262, 0.0036945, 0.0082473, -0.0043716, 0.0045195
9: -0.0050014, -0.0016747, -0.0050497, -0.0014452, -0.0035562, 0.0033750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0029421
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0029421
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0003234, 0.0020472, -0.0003612, 0.0019725, -0.0022958, 0.0024084
1: 0.9915439, 0.9965640, 0.9917022, 0.9966441, -0.0051001, 0.0048618
2: -0.0081676, -0.0042046, -0.0081866, -0.0044628, -0.0037048, 0.0039819
3: 0.0022835, 0.0052493, 0.0022362, 0.0051558, -0.0028723, 0.0030131
4: 0.0017401, 0.0066098, 0.0019127, 0.0063521, -0.0046120, 0.0046971
5: 0.0025960, 0.0082136, 0.0025064, 0.0080364, -0.0054404, 0.0057072
6: -0.0039873, 0.0014265, -0.0037849, 0.0015094, -0.0054967, 0.0052114
7: -0.0088155, -0.0064125, -0.0087397, -0.0063742, -0.0024413, 0.0023271
8: 0.0037760, 0.0082338, 0.0041152, 0.0082388, -0.0044343, 0.0041019
9: -0.0050237, -0.0015924, -0.0049155, -0.0015376, -0.0034861, 0.0033231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027740, upper bound: 0.0028236
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028464, upper bound: 0.0027933
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0003246, 0.0020649, -0.0002906, 0.0020367, -0.0023613, 0.0023555
1: 0.9915064, 0.9965667, 0.9915661, 0.9964945, -0.0049881, 0.0050005
2: -0.0081682, -0.0041435, -0.0081512, -0.0042409, -0.0039273, 0.0040078
3: 0.0022820, 0.0052715, 0.0023246, 0.0052362, -0.0029542, 0.0029469
4: 0.0016918, 0.0066708, 0.0017688, 0.0065736, -0.0048818, 0.0049020
5: 0.0025932, 0.0082556, 0.0026738, 0.0081887, -0.0055955, 0.0055817
6: -0.0040353, 0.0014292, -0.0039589, 0.0013546, -0.0053898, 0.0053880
7: -0.0088334, -0.0064113, -0.0088048, -0.0064458, -0.0023876, 0.0023935
8: 0.0036956, 0.0082339, 0.0038237, 0.0082294, -0.0045205, 0.0043905
9: -0.0050493, -0.0015906, -0.0050085, -0.0016399, -0.0034094, 0.0034179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029200, upper bound: 0.0028688
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029200, upper bound: 0.0030265
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0003372, 0.0020478, -0.0004287, 0.0019689, -0.0023061, 0.0024766
1: 0.9915425, 0.9965934, 0.9917098, 0.9967872, -0.0052447, 0.0048835
2: -0.0081746, -0.0042025, -0.0082204, -0.0044753, -0.0036993, 0.0040179
3: 0.0022662, 0.0052501, 0.0021517, 0.0051513, -0.0028851, 0.0030984
4: 0.0017384, 0.0066119, 0.0018025, 0.0063398, -0.0046013, 0.0048094
5: 0.0025633, 0.0082150, 0.0023463, 0.0080279, -0.0054647, 0.0058687
6: -0.0039890, 0.0014568, -0.0037751, 0.0016574, -0.0056464, 0.0052320
7: -0.0088161, -0.0063985, -0.0087360, -0.0063057, -0.0025103, 0.0023375
8: 0.0037732, 0.0082356, 0.0041315, 0.0082478, -0.0044390, 0.0040872
9: -0.0050246, -0.0015724, -0.0049103, -0.0014399, -0.0035847, 0.0033379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030059, upper bound: 0.0028582
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030081, upper bound: 0.0028370
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0003385, 0.0020656, -0.0003585, 0.0020362, -0.0023746, 0.0024241
1: 0.9915050, 0.9965960, 0.9915673, 0.9966386, -0.0051336, 0.0050287
2: -0.0081752, -0.0041413, -0.0081852, -0.0042427, -0.0039325, 0.0040439
3: 0.0022647, 0.0052722, 0.0022395, 0.0052355, -0.0029708, 0.0030327
4: 0.0016901, 0.0066729, 0.0017702, 0.0065718, -0.0048817, 0.0049027
5: 0.0025603, 0.0082570, 0.0025127, 0.0081875, -0.0056271, 0.0057443
6: -0.0040370, 0.0014595, -0.0039574, 0.0015036, -0.0055406, 0.0054170
7: -0.0088340, -0.0063972, -0.0088043, -0.0063769, -0.0024572, 0.0024070
8: 0.0036928, 0.0082358, 0.0038260, 0.0082385, -0.0045255, 0.0043888
9: -0.0050502, -0.0015705, -0.0050077, -0.0015415, -0.0035088, 0.0034372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030574, upper bound: 0.0028708
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030574, upper bound: 0.0030438
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003721, 0.0019461, -0.0003477, 0.0020480, -0.0024201, 0.0022938
1: 0.9917580, 0.9966673, 0.9915422, 0.9966156, -0.0048575, 0.0051250
2: -0.0081920, -0.0045540, -0.0081798, -0.0042020, -0.0039900, 0.0036258
3: 0.0022225, 0.0051228, 0.0022531, 0.0052503, -0.0030278, 0.0028696
4: 0.0018948, 0.0062612, 0.0017380, 0.0066124, -0.0047176, 0.0045232
5: 0.0024804, 0.0079739, 0.0025384, 0.0082154, -0.0057350, 0.0054355
6: -0.0037134, 0.0015334, -0.0039894, 0.0014798, -0.0051932, 0.0055228
7: -0.0087129, -0.0063631, -0.0088162, -0.0063879, -0.0023250, 0.0024531
8: 0.0042350, 0.0082403, 0.0037725, 0.0082370, -0.0039812, 0.0044271
9: -0.0048773, -0.0015218, -0.0050248, -0.0015572, -0.0033201, 0.0035030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028483, upper bound: 0.0027876
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027526, upper bound: 0.0026595
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003234, 0.0020373, -0.0003490, 0.0020642, -0.0023876, 0.0023863
1: 0.9915649, 0.9965641, 0.9915080, 0.9966183, -0.0050533, 0.0050561
2: -0.0081676, -0.0042389, -0.0081804, -0.0041461, -0.0040216, 0.0039416
3: 0.0022835, 0.0052369, 0.0022515, 0.0052705, -0.0029870, 0.0029854
4: 0.0017672, 0.0065756, 0.0016938, 0.0066682, -0.0049010, 0.0048818
5: 0.0025960, 0.0081901, 0.0025354, 0.0082538, -0.0056578, 0.0056547
6: -0.0039605, 0.0014266, -0.0040333, 0.0014826, -0.0054431, 0.0054598
7: -0.0088054, -0.0064125, -0.0088326, -0.0063866, -0.0024188, 0.0024202
8: 0.0038210, 0.0082338, 0.0036990, 0.0082372, -0.0043965, 0.0045121
9: -0.0050093, -0.0015923, -0.0050482, -0.0015553, -0.0034540, 0.0034559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0029047
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0030265
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003858, 0.0019467, -0.0004238, 0.0020491, -0.0024349, 0.0023705
1: 0.9917567, 0.9966961, 0.9915399, 0.9967766, -0.0050198, 0.0051562
2: -0.0081989, -0.0045518, -0.0082179, -0.0041982, -0.0040006, 0.0036661
3: 0.0022055, 0.0051236, 0.0021579, 0.0052516, -0.0030462, 0.0029657
4: 0.0018726, 0.0062634, 0.0017351, 0.0066162, -0.0047436, 0.0045283
5: 0.0024481, 0.0079754, 0.0023581, 0.0082180, -0.0057698, 0.0056173
6: -0.0037151, 0.0015633, -0.0039924, 0.0016465, -0.0053616, 0.0055556
7: -0.0087136, -0.0063493, -0.0088173, -0.0063108, -0.0024028, 0.0024681
8: 0.0042321, 0.0082421, 0.0037675, 0.0082472, -0.0039874, 0.0044342
9: -0.0048782, -0.0015020, -0.0050264, -0.0014470, -0.0034312, 0.0035243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029170, upper bound: 0.0026852
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029170, upper bound: 0.0026852
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003372, 0.0020379, -0.0004250, 0.0020652, -0.0024024, 0.0024630
1: 0.9915636, 0.9965933, 0.9915059, 0.9967793, -0.0052157, 0.0050874
2: -0.0081746, -0.0042367, -0.0082185, -0.0041426, -0.0040320, 0.0039817
3: 0.0022662, 0.0052377, 0.0021563, 0.0052718, -0.0030056, 0.0030813
4: 0.0017655, 0.0065777, 0.0016911, 0.0066717, -0.0049062, 0.0048866
5: 0.0025632, 0.0081916, 0.0023552, 0.0082561, -0.0056929, 0.0058364
6: -0.0039621, 0.0014569, -0.0040359, 0.0016492, -0.0056114, 0.0054928
7: -0.0088060, -0.0063985, -0.0088337, -0.0063095, -0.0024965, 0.0024352
8: 0.0038182, 0.0082356, 0.0036945, 0.0082473, -0.0044023, 0.0045189
9: -0.0050102, -0.0015723, -0.0050497, -0.0014452, -0.0035650, 0.0034774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030549, upper bound: 0.0029169
time: 1.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030549, upper bound: 0.0030436
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002908, 0.0020564, -0.0003183, 0.0019577, -0.0022485, 0.0023747
1: 0.9915244, 0.9964951, 0.9917334, 0.9965532, -0.0050288, 0.0047617
2: -0.0081514, -0.0041727, -0.0081651, -0.0045138, -0.0036375, 0.0039924
3: 0.0023243, 0.0052608, 0.0022898, 0.0051373, -0.0028130, 0.0029710
4: 0.0017149, 0.0066416, 0.0019826, 0.0063013, -0.0045864, 0.0046589
5: 0.0026732, 0.0082354, 0.0026081, 0.0080015, -0.0053283, 0.0056274
6: -0.0040123, 0.0013552, -0.0037449, 0.0014154, -0.0054277, 0.0051001
7: -0.0088248, -0.0064455, -0.0087247, -0.0064177, -0.0024071, 0.0022792
8: 0.0037341, 0.0082294, 0.0041822, 0.0082331, -0.0044634, 0.0040305
9: -0.0050370, -0.0016395, -0.0048941, -0.0015997, -0.0034373, 0.0032546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029315, upper bound: 0.0029515
time: 1.27 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028912, upper bound: 0.0028995
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002921, 0.0020715, -0.0002531, 0.0020312, -0.0023233, 0.0023246
1: 0.9914925, 0.9964977, 0.9915779, 0.9964153, -0.0049229, 0.0049198
2: -0.0081520, -0.0041208, -0.0081325, -0.0042600, -0.0038920, 0.0040117
3: 0.0023227, 0.0052797, 0.0023714, 0.0052293, -0.0029066, 0.0029083
4: 0.0016739, 0.0066934, 0.0017839, 0.0065546, -0.0048807, 0.0049095
5: 0.0026702, 0.0082711, 0.0027625, 0.0081756, -0.0055054, 0.0055086
6: -0.0040530, 0.0013579, -0.0039439, 0.0012726, -0.0053256, 0.0053019
7: -0.0088401, -0.0064442, -0.0087992, -0.0064838, -0.0023563, 0.0023550
8: 0.0036658, 0.0082296, 0.0038487, 0.0082244, -0.0045414, 0.0043629
9: -0.0050588, -0.0016376, -0.0050005, -0.0016941, -0.0033647, 0.0033629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029882, upper bound: 0.0029722
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029882, upper bound: 0.0029722
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003477, 0.0020480, -0.0003183, 0.0019577, -0.0023054, 0.0023663
1: 0.9915422, 0.9966156, 0.9917334, 0.9965532, -0.0050110, 0.0048822
2: -0.0081798, -0.0042020, -0.0081651, -0.0045138, -0.0036660, 0.0039631
3: 0.0022531, 0.0052503, 0.0022898, 0.0051373, -0.0028842, 0.0029604
4: 0.0017380, 0.0066124, 0.0019826, 0.0063013, -0.0045633, 0.0046298
5: 0.0025384, 0.0082154, 0.0026081, 0.0080015, -0.0054630, 0.0056073
6: -0.0039894, 0.0014798, -0.0037449, 0.0014154, -0.0054048, 0.0052247
7: -0.0088162, -0.0063879, -0.0087247, -0.0064177, -0.0023985, 0.0023368
8: 0.0037725, 0.0082370, 0.0041822, 0.0082331, -0.0044276, 0.0040353
9: -0.0050248, -0.0015572, -0.0048941, -0.0015997, -0.0034251, 0.0033370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029065, upper bound: 0.0028992
time: 1.52 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028715, upper bound: 0.0028481
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003490, 0.0020642, -0.0002531, 0.0020312, -0.0023802, 0.0023173
1: 0.9915080, 0.9966183, 0.9915779, 0.9964153, -0.0049073, 0.0050404
2: -0.0081804, -0.0041461, -0.0081325, -0.0042600, -0.0039205, 0.0039865
3: 0.0022515, 0.0052705, 0.0023714, 0.0052293, -0.0029778, 0.0028991
4: 0.0016938, 0.0066682, 0.0017839, 0.0065546, -0.0048607, 0.0048844
5: 0.0025354, 0.0082538, 0.0027625, 0.0081756, -0.0056402, 0.0054913
6: -0.0040333, 0.0014826, -0.0039439, 0.0012726, -0.0053058, 0.0054265
7: -0.0088326, -0.0063866, -0.0087992, -0.0064838, -0.0023489, 0.0024126
8: 0.0036990, 0.0082372, 0.0038487, 0.0082244, -0.0045114, 0.0043653
9: -0.0050482, -0.0015553, -0.0050005, -0.0016941, -0.0033542, 0.0034452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029211, upper bound: 0.0028325
time: 1.21 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029211, upper bound: 0.0030560
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003588, 0.0020549, -0.0003316, 0.0019584, -0.0023172, 0.0023865
1: 0.9915276, 0.9966391, 0.9917320, 0.9965814, -0.0050538, 0.0049071
2: -0.0081854, -0.0041781, -0.0081717, -0.0045115, -0.0036739, 0.0039937
3: 0.0022392, 0.0052589, 0.0022733, 0.0051382, -0.0028990, 0.0029857
4: 0.0017191, 0.0066363, 0.0019610, 0.0063036, -0.0045845, 0.0046753
5: 0.0025120, 0.0082318, 0.0025766, 0.0080031, -0.0054910, 0.0056552
6: -0.0040081, 0.0015042, -0.0037467, 0.0014445, -0.0054526, 0.0052509
7: -0.0088232, -0.0063766, -0.0087254, -0.0064042, -0.0024190, 0.0023488
8: 0.0037411, 0.0082385, 0.0041792, 0.0082349, -0.0044583, 0.0040367
9: -0.0050348, -0.0015411, -0.0048951, -0.0015805, -0.0034543, 0.0033540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0030807
time: 1.24 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0031177
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003601, 0.0020702, -0.0002665, 0.0020318, -0.0023919, 0.0023367
1: 0.9914953, 0.9966418, 0.9915764, 0.9964435, -0.0049483, 0.0050654
2: -0.0081860, -0.0041252, -0.0081392, -0.0042578, -0.0039282, 0.0040140
3: 0.0022376, 0.0052781, 0.0023547, 0.0052301, -0.0029924, 0.0029234
4: 0.0016774, 0.0066890, 0.0017821, 0.0065567, -0.0048794, 0.0049069
5: 0.0025090, 0.0082681, 0.0027308, 0.0081771, -0.0056681, 0.0055373
6: -0.0040496, 0.0015070, -0.0039456, 0.0013019, -0.0053515, 0.0054526
7: -0.0088388, -0.0063753, -0.0087999, -0.0064702, -0.0023686, 0.0024245
8: 0.0036716, 0.0082387, 0.0038458, 0.0082262, -0.0045372, 0.0043687
9: -0.0050570, -0.0015392, -0.0050014, -0.0016747, -0.0033823, 0.0034622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031321
time: 1.00 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031455
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004238, 0.0020491, -0.0003316, 0.0019584, -0.0023822, 0.0023806
1: 0.9915399, 0.9967766, 0.9917320, 0.9965814, -0.0050415, 0.0050446
2: -0.0082179, -0.0041982, -0.0081717, -0.0045115, -0.0037064, 0.0039735
3: 0.0021579, 0.0052516, 0.0022733, 0.0051382, -0.0029802, 0.0029784
4: 0.0017351, 0.0066162, 0.0019610, 0.0063036, -0.0045685, 0.0046552
5: 0.0023581, 0.0082180, 0.0025766, 0.0080031, -0.0056449, 0.0056414
6: -0.0039924, 0.0016465, -0.0037467, 0.0014445, -0.0054368, 0.0053932
7: -0.0088173, -0.0063108, -0.0087254, -0.0064042, -0.0024131, 0.0024147
8: 0.0037675, 0.0082472, 0.0041792, 0.0082349, -0.0044350, 0.0040420
9: -0.0050264, -0.0014470, -0.0048951, -0.0015805, -0.0034459, 0.0034481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028431, upper bound: 0.0030336
time: 1.31 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028431, upper bound: 0.0030722
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004250, 0.0020652, -0.0002665, 0.0020318, -0.0024569, 0.0023317
1: 0.9915059, 0.9967793, 0.9915764, 0.9964435, -0.0049376, 0.0052028
2: -0.0082185, -0.0041426, -0.0081392, -0.0042578, -0.0039607, 0.0039966
3: 0.0021563, 0.0052718, 0.0023547, 0.0052301, -0.0030737, 0.0029171
4: 0.0016911, 0.0066717, 0.0017821, 0.0065567, -0.0048656, 0.0048895
5: 0.0023552, 0.0082561, 0.0027308, 0.0081771, -0.0058220, 0.0055254
6: -0.0040359, 0.0016492, -0.0039456, 0.0013019, -0.0053379, 0.0055949
7: -0.0088337, -0.0063095, -0.0087999, -0.0064702, -0.0023635, 0.0024904
8: 0.0036945, 0.0082473, 0.0038458, 0.0082262, -0.0045195, 0.0043716
9: -0.0050497, -0.0014452, -0.0050014, -0.0016747, -0.0033750, 0.0035562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031211
time: 1.49 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031344
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0003612, 0.0019725, -0.0003234, 0.0020472, -0.0024084, 0.0022958
1: 0.9917022, 0.9966441, 0.9915439, 0.9965640, -0.0048618, 0.0051001
2: -0.0081866, -0.0044628, -0.0081676, -0.0042046, -0.0039819, 0.0037048
3: 0.0022362, 0.0051558, 0.0022835, 0.0052493, -0.0030131, 0.0028723
4: 0.0019127, 0.0063521, 0.0017401, 0.0066098, -0.0046971, 0.0046120
5: 0.0025064, 0.0080364, 0.0025960, 0.0082136, -0.0057072, 0.0054404
6: -0.0037849, 0.0015094, -0.0039873, 0.0014265, -0.0052114, 0.0054967
7: -0.0087397, -0.0063742, -0.0088155, -0.0064125, -0.0023271, 0.0024413
8: 0.0041152, 0.0082388, 0.0037760, 0.0082338, -0.0041019, 0.0044343
9: -0.0049155, -0.0015376, -0.0050237, -0.0015924, -0.0033231, 0.0034861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028236, upper bound: 0.0028503
time: 1.33 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027933, upper bound: 0.0028464
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002906, 0.0020367, -0.0003246, 0.0020649, -0.0023555, 0.0023613
1: 0.9915661, 0.9964945, 0.9915064, 0.9965667, -0.0050005, 0.0049881
2: -0.0081512, -0.0042409, -0.0081682, -0.0041435, -0.0040078, 0.0039273
3: 0.0023246, 0.0052362, 0.0022820, 0.0052715, -0.0029469, 0.0029542
4: 0.0017688, 0.0065736, 0.0016918, 0.0066708, -0.0049020, 0.0048818
5: 0.0026738, 0.0081887, 0.0025932, 0.0082556, -0.0055817, 0.0055955
6: -0.0039589, 0.0013546, -0.0040353, 0.0014292, -0.0053880, 0.0053898
7: -0.0088048, -0.0064458, -0.0088334, -0.0064113, -0.0023935, 0.0023876
8: 0.0038237, 0.0082294, 0.0036956, 0.0082339, -0.0043905, 0.0045205
9: -0.0050085, -0.0016399, -0.0050493, -0.0015906, -0.0034179, 0.0034094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028688, upper bound: 0.0029200
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028688, upper bound: 0.0030575
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004287, 0.0019689, -0.0003372, 0.0020478, -0.0024766, 0.0023061
1: 0.9917098, 0.9967872, 0.9915425, 0.9965934, -0.0048835, 0.0052447
2: -0.0082204, -0.0044753, -0.0081746, -0.0042025, -0.0040179, 0.0036993
3: 0.0021517, 0.0051513, 0.0022662, 0.0052501, -0.0030984, 0.0028851
4: 0.0018025, 0.0063398, 0.0017384, 0.0066119, -0.0048094, 0.0046013
5: 0.0023463, 0.0080279, 0.0025633, 0.0082150, -0.0058687, 0.0054647
6: -0.0037751, 0.0016574, -0.0039890, 0.0014568, -0.0052320, 0.0056464
7: -0.0087360, -0.0063057, -0.0088161, -0.0063985, -0.0023375, 0.0025103
8: 0.0041315, 0.0082478, 0.0037732, 0.0082356, -0.0040872, 0.0044390
9: -0.0049103, -0.0014399, -0.0050246, -0.0015724, -0.0033379, 0.0035847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028582, upper bound: 0.0030059
time: 1.18 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028370, upper bound: 0.0030081
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0003585, 0.0020362, -0.0003385, 0.0020656, -0.0024241, 0.0023746
1: 0.9915673, 0.9966386, 0.9915050, 0.9965960, -0.0050287, 0.0051336
2: -0.0081852, -0.0042427, -0.0081752, -0.0041413, -0.0040439, 0.0039325
3: 0.0022395, 0.0052355, 0.0022647, 0.0052722, -0.0030327, 0.0029708
4: 0.0017702, 0.0065718, 0.0016901, 0.0066729, -0.0049027, 0.0048817
5: 0.0025127, 0.0081875, 0.0025603, 0.0082570, -0.0057443, 0.0056271
6: -0.0039574, 0.0015036, -0.0040370, 0.0014595, -0.0054170, 0.0055406
7: -0.0088043, -0.0063769, -0.0088340, -0.0063972, -0.0024070, 0.0024572
8: 0.0038260, 0.0082385, 0.0036928, 0.0082358, -0.0043888, 0.0045255
9: -0.0050077, -0.0015415, -0.0050502, -0.0015705, -0.0034372, 0.0035088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0030574
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0031439
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003477, 0.0020480, -0.0003721, 0.0019461, -0.0022938, 0.0024201
1: 0.9915422, 0.9966156, 0.9917580, 0.9966673, -0.0051250, 0.0048575
2: -0.0081798, -0.0042020, -0.0081920, -0.0045540, -0.0036258, 0.0039900
3: 0.0022531, 0.0052503, 0.0022225, 0.0051228, -0.0028696, 0.0030278
4: 0.0017380, 0.0066124, 0.0018948, 0.0062612, -0.0045232, 0.0047176
5: 0.0025384, 0.0082154, 0.0024804, 0.0079739, -0.0054355, 0.0057350
6: -0.0039894, 0.0014798, -0.0037134, 0.0015334, -0.0055228, 0.0051932
7: -0.0088162, -0.0063879, -0.0087129, -0.0063631, -0.0024531, 0.0023250
8: 0.0037725, 0.0082370, 0.0042350, 0.0082403, -0.0044271, 0.0039812
9: -0.0050248, -0.0015572, -0.0048773, -0.0015218, -0.0035030, 0.0033201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027899, upper bound: 0.0028483
time: 1.33 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026595, upper bound: 0.0027526
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003490, 0.0020642, -0.0003234, 0.0020373, -0.0023863, 0.0023876
1: 0.9915080, 0.9966183, 0.9915649, 0.9965641, -0.0050561, 0.0050533
2: -0.0081804, -0.0041461, -0.0081676, -0.0042389, -0.0039416, 0.0040216
3: 0.0022515, 0.0052705, 0.0022835, 0.0052369, -0.0029854, 0.0029870
4: 0.0016938, 0.0066682, 0.0017672, 0.0065756, -0.0048818, 0.0049010
5: 0.0025354, 0.0082538, 0.0025960, 0.0081901, -0.0056547, 0.0056578
6: -0.0040333, 0.0014826, -0.0039605, 0.0014266, -0.0054598, 0.0054431
7: -0.0088326, -0.0063866, -0.0088054, -0.0064125, -0.0024202, 0.0024188
8: 0.0036990, 0.0082372, 0.0038210, 0.0082338, -0.0045121, 0.0043965
9: -0.0050482, -0.0015553, -0.0050093, -0.0015923, -0.0034559, 0.0034540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029047, upper bound: 0.0028325
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029047, upper bound: 0.0030545
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004238, 0.0020491, -0.0003858, 0.0019467, -0.0023705, 0.0024349
1: 0.9915399, 0.9967766, 0.9917567, 0.9966961, -0.0051562, 0.0050198
2: -0.0082179, -0.0041982, -0.0081989, -0.0045518, -0.0036661, 0.0040006
3: 0.0021579, 0.0052516, 0.0022055, 0.0051236, -0.0029657, 0.0030462
4: 0.0017351, 0.0066162, 0.0018726, 0.0062634, -0.0045283, 0.0047436
5: 0.0023581, 0.0082180, 0.0024481, 0.0079754, -0.0056173, 0.0057698
6: -0.0039924, 0.0016465, -0.0037151, 0.0015633, -0.0055556, 0.0053616
7: -0.0088173, -0.0063108, -0.0087136, -0.0063493, -0.0024681, 0.0024028
8: 0.0037675, 0.0082472, 0.0042321, 0.0082421, -0.0044342, 0.0039874
9: -0.0050264, -0.0014470, -0.0048782, -0.0015020, -0.0035243, 0.0034312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030037
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030484
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004250, 0.0020652, -0.0003372, 0.0020379, -0.0024630, 0.0024024
1: 0.9915059, 0.9967793, 0.9915636, 0.9965933, -0.0050874, 0.0052157
2: -0.0082185, -0.0041426, -0.0081746, -0.0042367, -0.0039817, 0.0040320
3: 0.0021563, 0.0052718, 0.0022662, 0.0052377, -0.0030813, 0.0030056
4: 0.0016911, 0.0066717, 0.0017655, 0.0065777, -0.0048866, 0.0049062
5: 0.0023552, 0.0082561, 0.0025632, 0.0081916, -0.0058364, 0.0056929
6: -0.0040359, 0.0016492, -0.0039621, 0.0014569, -0.0054928, 0.0056114
7: -0.0088337, -0.0063095, -0.0088060, -0.0063985, -0.0024352, 0.0024965
8: 0.0036945, 0.0082473, 0.0038182, 0.0082356, -0.0045189, 0.0044023
9: -0.0050497, -0.0014452, -0.0050102, -0.0015723, -0.0034774, 0.0035650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0031211
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0031344
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0003612, 0.0019725, -0.0003734, 0.0020572, -0.0024184, 0.0023459
1: 0.9917022, 0.9966441, 0.9915228, 0.9966700, -0.0049678, 0.0051213
2: -0.0081866, -0.0044628, -0.0081927, -0.0041702, -0.0040164, 0.0037298
3: 0.0022362, 0.0051558, 0.0022209, 0.0052618, -0.0030256, 0.0029349
4: 0.0019127, 0.0063521, 0.0017129, 0.0066442, -0.0047315, 0.0046393
5: 0.0025064, 0.0080364, 0.0024774, 0.0082372, -0.0057308, 0.0055590
6: -0.0037849, 0.0015094, -0.0040144, 0.0015362, -0.0053210, 0.0055238
7: -0.0087397, -0.0063742, -0.0088256, -0.0063618, -0.0023779, 0.0024514
8: 0.0041152, 0.0082388, 0.0037307, 0.0082404, -0.0041040, 0.0044707
9: -0.0049155, -0.0015376, -0.0050381, -0.0015200, -0.0033955, 0.0035005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029945, upper bound: 0.0029321
time: 1.66 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029876, upper bound: 0.0029242
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002906, 0.0020367, -0.0003747, 0.0020725, -0.0023631, 0.0024114
1: 0.9915661, 0.9964945, 0.9914904, 0.9966727, -0.0051066, 0.0050040
2: -0.0081512, -0.0042409, -0.0081933, -0.0041173, -0.0040339, 0.0039524
3: 0.0023246, 0.0052362, 0.0022193, 0.0052809, -0.0029563, 0.0030169
4: 0.0017688, 0.0065736, 0.0016711, 0.0066969, -0.0049281, 0.0049025
5: 0.0026738, 0.0081887, 0.0024744, 0.0082735, -0.0055997, 0.0057143
6: -0.0039589, 0.0013546, -0.0040558, 0.0015390, -0.0054978, 0.0054104
7: -0.0088048, -0.0064458, -0.0088411, -0.0063605, -0.0024443, 0.0023953
8: 0.0038237, 0.0082294, 0.0036612, 0.0082406, -0.0043935, 0.0045480
9: -0.0050085, -0.0016399, -0.0050603, -0.0015181, -0.0034904, 0.0034204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030623, upper bound: 0.0030147
time: 1.32 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030623, upper bound: 0.0030816
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004287, 0.0019689, -0.0003865, 0.0020579, -0.0024866, 0.0023553
1: 0.9917098, 0.9967872, 0.9915214, 0.9966977, -0.0049879, 0.0052658
2: -0.0082204, -0.0044753, -0.0081992, -0.0041679, -0.0040525, 0.0037240
3: 0.0021517, 0.0051513, 0.0022046, 0.0052626, -0.0031109, 0.0029467
4: 0.0018025, 0.0063398, 0.0017111, 0.0066464, -0.0048439, 0.0046287
5: 0.0023463, 0.0080279, 0.0024465, 0.0082388, -0.0058925, 0.0055814
6: -0.0037751, 0.0016574, -0.0040161, 0.0015648, -0.0053399, 0.0056735
7: -0.0087360, -0.0063057, -0.0088262, -0.0063486, -0.0023875, 0.0025205
8: 0.0041315, 0.0082478, 0.0037277, 0.0082422, -0.0040890, 0.0044758
9: -0.0049103, -0.0014399, -0.0050391, -0.0015010, -0.0034092, 0.0035992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030201, upper bound: 0.0030782
time: 1.21 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030127, upper bound: 0.0030793
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0003585, 0.0020362, -0.0003878, 0.0020732, -0.0024317, 0.0024239
1: 0.9915673, 0.9966386, 0.9914889, 0.9967003, -0.0051330, 0.0051497
2: -0.0081852, -0.0042427, -0.0081998, -0.0041150, -0.0040702, 0.0039571
3: 0.0022395, 0.0052355, 0.0022030, 0.0052818, -0.0030422, 0.0030325
4: 0.0017702, 0.0065718, 0.0016693, 0.0066992, -0.0049289, 0.0049025
5: 0.0025127, 0.0081875, 0.0024435, 0.0082751, -0.0057624, 0.0057440
6: -0.0039574, 0.0015036, -0.0040576, 0.0015676, -0.0055250, 0.0055612
7: -0.0088043, -0.0063769, -0.0088418, -0.0063473, -0.0024570, 0.0024649
8: 0.0038260, 0.0082385, 0.0036583, 0.0082424, -0.0043922, 0.0045535
9: -0.0050077, -0.0015415, -0.0050612, -0.0014992, -0.0035085, 0.0035198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030789, upper bound: 0.0031310
time: 1.31 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030789, upper bound: 0.0031545
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004099, 0.0019551, -0.0003734, 0.0020572, -0.0024671, 0.0023285
1: 0.9917390, 0.9967474, 0.9915228, 0.9966700, -0.0049310, 0.0052246
2: -0.0082109, -0.0045228, -0.0081927, -0.0041702, -0.0040408, 0.0036699
3: 0.0021752, 0.0051341, 0.0022209, 0.0052618, -0.0030866, 0.0029132
4: 0.0018332, 0.0062923, 0.0017129, 0.0066442, -0.0048109, 0.0045795
5: 0.0023910, 0.0079953, 0.0024774, 0.0082372, -0.0058463, 0.0055179
6: -0.0037378, 0.0016162, -0.0040144, 0.0015362, -0.0052740, 0.0056305
7: -0.0087221, -0.0063248, -0.0088256, -0.0063618, -0.0023603, 0.0025008
8: 0.0041940, 0.0082453, 0.0037307, 0.0082404, -0.0040278, 0.0044737
9: -0.0048904, -0.0014671, -0.0050381, -0.0015200, -0.0033704, 0.0035710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029424, upper bound: 0.0027979
time: 1.34 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029308, upper bound: 0.0027466
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003476, 0.0020330, -0.0003747, 0.0020725, -0.0024201, 0.0024077
1: 0.9915740, 0.9966153, 0.9914904, 0.9966727, -0.0050987, 0.0051249
2: -0.0081797, -0.0042538, -0.0081933, -0.0041173, -0.0040625, 0.0039395
3: 0.0022532, 0.0052315, 0.0022193, 0.0052809, -0.0030277, 0.0030122
4: 0.0017790, 0.0065607, 0.0016711, 0.0066969, -0.0049179, 0.0048896
5: 0.0025386, 0.0081798, 0.0024744, 0.0082735, -0.0057349, 0.0057054
6: -0.0039487, 0.0014796, -0.0040558, 0.0015390, -0.0054877, 0.0055354
7: -0.0088010, -0.0063880, -0.0088411, -0.0063605, -0.0024405, 0.0024531
8: 0.0038406, 0.0082370, 0.0036612, 0.0082406, -0.0043795, 0.0045538
9: -0.0050031, -0.0015573, -0.0050603, -0.0015181, -0.0034850, 0.0035030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030479, upper bound: 0.0029783
time: 1.35 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030479, upper bound: 0.0030770
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004814, 0.0019557, -0.0003865, 0.0020579, -0.0025393, 0.0023421
1: 0.9917378, 0.9968988, 0.9915214, 0.9966977, -0.0049599, 0.0053774
2: -0.0082467, -0.0045209, -0.0081992, -0.0041679, -0.0040788, 0.0036783
3: 0.0020858, 0.0051348, 0.0022046, 0.0052626, -0.0031768, 0.0029302
4: 0.0017166, 0.0062943, 0.0017111, 0.0066464, -0.0049298, 0.0045832
5: 0.0022215, 0.0079966, 0.0024465, 0.0082388, -0.0060173, 0.0055501
6: -0.0037393, 0.0017729, -0.0040161, 0.0015648, -0.0053041, 0.0057890
7: -0.0087227, -0.0062523, -0.0088262, -0.0063486, -0.0023741, 0.0025739
8: 0.0041915, 0.0082549, 0.0037277, 0.0082422, -0.0040331, 0.0044796
9: -0.0048912, -0.0013636, -0.0050391, -0.0015010, -0.0033901, 0.0036755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029716, upper bound: 0.0029920
time: 1.25 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029625, upper bound: 0.0029833
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004237, 0.0020355, -0.0003878, 0.0020732, -0.0024969, 0.0024233
1: 0.9915687, 0.9967764, 0.9914889, 0.9967003, -0.0051316, 0.0052875
2: -0.0082178, -0.0042450, -0.0081998, -0.0041150, -0.0041028, 0.0039548
3: 0.0021580, 0.0052347, 0.0022030, 0.0052818, -0.0031238, 0.0030317
4: 0.0017721, 0.0065694, 0.0016693, 0.0066992, -0.0049271, 0.0049001
5: 0.0023583, 0.0081859, 0.0024435, 0.0082751, -0.0059168, 0.0057424
6: -0.0039556, 0.0016464, -0.0040576, 0.0015676, -0.0055232, 0.0057040
7: -0.0088036, -0.0063108, -0.0088418, -0.0063473, -0.0024563, 0.0025309
8: 0.0038291, 0.0082472, 0.0036583, 0.0082424, -0.0043940, 0.0045599
9: -0.0050068, -0.0014471, -0.0050612, -0.0014992, -0.0035076, 0.0036141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030624, upper bound: 0.0030849
time: 1.38 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030624, upper bound: 0.0031484
time: 1.54 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0003734, 0.0020572, -0.0004099, 0.0019551, -0.0023285, 0.0024671
1: 0.9915228, 0.9966700, 0.9917390, 0.9967474, -0.0052246, 0.0049310
2: -0.0081927, -0.0041702, -0.0082109, -0.0045228, -0.0036699, 0.0040408
3: 0.0022209, 0.0052618, 0.0021752, 0.0051341, -0.0029132, 0.0030866
4: 0.0017129, 0.0066442, 0.0018332, 0.0062923, -0.0045795, 0.0048109
5: 0.0024774, 0.0082372, 0.0023910, 0.0079953, -0.0055179, 0.0058463
6: -0.0040144, 0.0015362, -0.0037378, 0.0016162, -0.0056305, 0.0052740
7: -0.0088256, -0.0063618, -0.0087221, -0.0063248, -0.0025008, 0.0023603
8: 0.0037307, 0.0082404, 0.0041940, 0.0082453, -0.0044737, 0.0040278
9: -0.0050381, -0.0015200, -0.0048904, -0.0014671, -0.0035710, 0.0033704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027956, upper bound: 0.0029982
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027466, upper bound: 0.0029584
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0003747, 0.0020725, -0.0003476, 0.0020330, -0.0024077, 0.0024201
1: 0.9914904, 0.9966727, 0.9915740, 0.9966153, -0.0051249, 0.0050987
2: -0.0081933, -0.0041173, -0.0081797, -0.0042538, -0.0039395, 0.0040625
3: 0.0022193, 0.0052809, 0.0022532, 0.0052315, -0.0030122, 0.0030277
4: 0.0016711, 0.0066969, 0.0017790, 0.0065607, -0.0048896, 0.0049179
5: 0.0024744, 0.0082735, 0.0025386, 0.0081798, -0.0057054, 0.0057349
6: -0.0040558, 0.0015390, -0.0039487, 0.0014796, -0.0055354, 0.0054877
7: -0.0088411, -0.0063605, -0.0088010, -0.0063880, -0.0024531, 0.0024405
8: 0.0036612, 0.0082406, 0.0038406, 0.0082370, -0.0045538, 0.0043795
9: -0.0050603, -0.0015181, -0.0050031, -0.0015573, -0.0035030, 0.0034850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029257, upper bound: 0.0030904
time: 1.22 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029257, upper bound: 0.0031549
time: 1.72 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0003865, 0.0020579, -0.0004814, 0.0019557, -0.0023421, 0.0025393
1: 0.9915214, 0.9966977, 0.9917378, 0.9968988, -0.0053774, 0.0049599
2: -0.0081992, -0.0041679, -0.0082467, -0.0045209, -0.0036783, 0.0040788
3: 0.0022046, 0.0052626, 0.0020858, 0.0051348, -0.0029302, 0.0031768
4: 0.0017111, 0.0066464, 0.0017166, 0.0062943, -0.0045832, 0.0049298
5: 0.0024465, 0.0082388, 0.0022215, 0.0079966, -0.0055501, 0.0060173
6: -0.0040161, 0.0015648, -0.0037393, 0.0017729, -0.0057890, 0.0053041
7: -0.0088262, -0.0063486, -0.0087227, -0.0062523, -0.0025739, 0.0023741
8: 0.0037277, 0.0082422, 0.0041915, 0.0082549, -0.0044796, 0.0040331
9: -0.0050391, -0.0015010, -0.0048912, -0.0013636, -0.0036755, 0.0033901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029668, upper bound: 0.0030297
time: 1.29 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029551, upper bound: 0.0030001
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0003878, 0.0020732, -0.0004237, 0.0020355, -0.0024233, 0.0024969
1: 0.9914889, 0.9967003, 0.9915687, 0.9967764, -0.0052875, 0.0051316
2: -0.0081998, -0.0041150, -0.0082178, -0.0042450, -0.0039548, 0.0041028
3: 0.0022030, 0.0052818, 0.0021580, 0.0052347, -0.0030317, 0.0031238
4: 0.0016693, 0.0066992, 0.0017721, 0.0065694, -0.0049001, 0.0049271
5: 0.0024435, 0.0082751, 0.0023583, 0.0081859, -0.0057424, 0.0059168
6: -0.0040576, 0.0015676, -0.0039556, 0.0016464, -0.0057040, 0.0055232
7: -0.0088418, -0.0063473, -0.0088036, -0.0063108, -0.0025309, 0.0024563
8: 0.0036583, 0.0082424, 0.0038291, 0.0082472, -0.0045599, 0.0043940
9: -0.0050612, -0.0014992, -0.0050068, -0.0014471, -0.0036141, 0.0035076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030189, upper bound: 0.0031082
time: 1.34 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030189, upper bound: 0.0031615
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004099, 0.0019551, -0.0004349, 0.0020516, -0.0024615, 0.0023900
1: 0.9917390, 0.9967474, 0.9915346, 0.9968003, -0.0050613, 0.0052128
2: -0.0082109, -0.0045228, -0.0082235, -0.0041894, -0.0040215, 0.0037006
3: 0.0021752, 0.0051341, 0.0021439, 0.0052548, -0.0030796, 0.0029902
4: 0.0018332, 0.0062923, 0.0017281, 0.0066249, -0.0047917, 0.0045642
5: 0.0023910, 0.0079953, 0.0023316, 0.0082240, -0.0058331, 0.0056637
6: -0.0037378, 0.0016162, -0.0039992, 0.0016710, -0.0054088, 0.0056154
7: -0.0087221, -0.0063248, -0.0088199, -0.0062994, -0.0024227, 0.0024951
8: 0.0041940, 0.0082453, 0.0037560, 0.0082487, -0.0040262, 0.0044462
9: -0.0048904, -0.0014671, -0.0050301, -0.0014309, -0.0034595, 0.0035630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029287, upper bound: 0.0027978
time: 1.39 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029170, upper bound: 0.0027466
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003476, 0.0020330, -0.0004362, 0.0020677, -0.0024153, 0.0024692
1: 0.9915740, 0.9966153, 0.9915005, 0.9968030, -0.0052289, 0.0051148
2: -0.0081797, -0.0042538, -0.0082241, -0.0041338, -0.0040460, 0.0039703
3: 0.0022532, 0.0052315, 0.0021424, 0.0052750, -0.0030217, 0.0030891
4: 0.0017790, 0.0065607, 0.0016841, 0.0066805, -0.0049015, 0.0048766
5: 0.0025386, 0.0081798, 0.0023287, 0.0082622, -0.0057236, 0.0058512
6: -0.0039487, 0.0014796, -0.0040429, 0.0016737, -0.0056225, 0.0055225
7: -0.0088010, -0.0063880, -0.0088363, -0.0062982, -0.0025029, 0.0024483
8: 0.0038406, 0.0082370, 0.0036829, 0.0082488, -0.0043805, 0.0045284
9: -0.0050031, -0.0015573, -0.0050534, -0.0014291, -0.0035740, 0.0034961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030017, upper bound: 0.0029476
time: 1.32 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030017, upper bound: 0.0030764
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004814, 0.0019557, -0.0004488, 0.0020523, -0.0025337, 0.0024044
1: 0.9917378, 0.9968988, 0.9915332, 0.9968296, -0.0050917, 0.0053656
2: -0.0082467, -0.0045209, -0.0082304, -0.0041872, -0.0040595, 0.0037095
3: 0.0020858, 0.0051348, 0.0021266, 0.0052556, -0.0031699, 0.0030081
4: 0.0017166, 0.0062943, 0.0017263, 0.0066272, -0.0049106, 0.0045679
5: 0.0022215, 0.0079966, 0.0022989, 0.0082256, -0.0060041, 0.0056978
6: -0.0037393, 0.0017729, -0.0040010, 0.0017013, -0.0054406, 0.0057739
7: -0.0087227, -0.0062523, -0.0088206, -0.0062854, -0.0024372, 0.0025683
8: 0.0041915, 0.0082549, 0.0037531, 0.0082505, -0.0040312, 0.0044521
9: -0.0048912, -0.0013636, -0.0050310, -0.0014109, -0.0034803, 0.0036674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0029920
time: 1.35 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029507, upper bound: 0.0029833
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004237, 0.0020355, -0.0004500, 0.0020684, -0.0024921, 0.0024855
1: 0.9915687, 0.9967764, 0.9914991, 0.9968323, -0.0052636, 0.0052773
2: -0.0082178, -0.0042450, -0.0082310, -0.0041315, -0.0040863, 0.0039860
3: 0.0021580, 0.0052347, 0.0021251, 0.0052758, -0.0031178, 0.0031096
4: 0.0017721, 0.0065694, 0.0016823, 0.0066827, -0.0049107, 0.0048871
5: 0.0023583, 0.0081859, 0.0022959, 0.0082637, -0.0059055, 0.0058900
6: -0.0039556, 0.0016464, -0.0040447, 0.0017040, -0.0056597, 0.0056910
7: -0.0088036, -0.0063108, -0.0088369, -0.0062842, -0.0025194, 0.0025261
8: 0.0038291, 0.0082472, 0.0036799, 0.0082507, -0.0043937, 0.0045345
9: -0.0050068, -0.0014471, -0.0050543, -0.0014091, -0.0035977, 0.0036072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030246, upper bound: 0.0030631
time: 1.28 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030246, upper bound: 0.0031484
time: 1.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.08 seconds
IS_A1_B1_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028976, upper bound: 0.0027993
IS_A1_B1_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028691, upper bound: 0.0027149
IS_A1_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030101, upper bound: 0.0028998
IS_A1_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030101, upper bound: 0.0029421
IS_A1_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029298, upper bound: 0.0029546
IS_A1_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029102, upper bound: 0.0029012
IS_A1_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030382, upper bound: 0.0030268
IS_A1_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030382, upper bound: 0.0030710
IS_A1_B1_A1_B2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0026397, upper bound: 0.0028039
IS_A1_B1_A1_B2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0023703, upper bound: 0.0026745
IS_A1_B1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028536, upper bound: 0.0029916
IS_A1_B1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028536, upper bound: 0.0030468
IS_A1_B1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028089, upper bound: 0.0028548
IS_A1_B1_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0026595, upper bound: 0.0028023
IS_A1_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029608, upper bound: 0.0030167
IS_A1_B1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029608, upper bound: 0.0030626
IS_A1_B1_A2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028039, upper bound: 0.0026397
IS_A1_B1_A2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0026745, upper bound: 0.0023703
IS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0028536
IS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029916, upper bound: 0.0029169
IS_A1_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028548, upper bound: 0.0028088
IS_A1_B1_A2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028022, upper bound: 0.0026595
IS_A1_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030167, upper bound: 0.0029608
IS_A1_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030167, upper bound: 0.0030441
IS_A1_B1_A2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027812, upper bound: 0.0026396
IS_A1_B1_A2_B2_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0026543, upper bound: 0.0023703
IS_A1_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028696, upper bound: 0.0028240
IS_A1_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028696, upper bound: 0.0028240
IS_A1_B1_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028262, upper bound: 0.0028087
IS_A1_B1_A2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027764, upper bound: 0.0026595
IS_A1_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0029410
IS_A1_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0030437
IS_A1_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029515, upper bound: 0.0029315
IS_A1_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028995, upper bound: 0.0028912
IS_A1_B2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029722, upper bound: 0.0029882
IS_A1_B2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029722, upper bound: 0.0030491
IS_A1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028992, upper bound: 0.0029065
IS_A1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028481, upper bound: 0.0028715
IS_A1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0029211
IS_A1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0030460
IS_A1_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029981, upper bound: 0.0028433
IS_A1_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029981, upper bound: 0.0028433
IS_A1_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030628, upper bound: 0.0029421
IS_A1_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030628, upper bound: 0.0030662
IS_A1_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0028431
IS_A1_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0030166
IS_A1_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0029421
IS_A1_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030560, upper bound: 0.0029421
IS_A1_B2_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027740, upper bound: 0.0028236
IS_A1_B2_A2_B1_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028464, upper bound: 0.0027933
IS_A1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029200, upper bound: 0.0028688
IS_A1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029200, upper bound: 0.0030265
IS_A1_B2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030059, upper bound: 0.0028582
IS_A1_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030081, upper bound: 0.0028370
IS_A1_B2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030574, upper bound: 0.0028708
IS_A1_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030574, upper bound: 0.0030438
IS_A1_B2_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028483, upper bound: 0.0027876
IS_A1_B2_A2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027526, upper bound: 0.0026595
IS_A1_B2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0029047
IS_A1_B2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028325, upper bound: 0.0030265
IS_A1_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029170, upper bound: 0.0026852
IS_A1_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029170, upper bound: 0.0026852
IS_A1_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030549, upper bound: 0.0029169
IS_A1_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030549, upper bound: 0.0030436
IS_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029315, upper bound: 0.0029515
IS_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028912, upper bound: 0.0028995
IS_A2_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029882, upper bound: 0.0029722
IS_A2_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029882, upper bound: 0.0029722
IS_A2_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029065, upper bound: 0.0028992
IS_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028715, upper bound: 0.0028481
IS_A2_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029211, upper bound: 0.0028325
IS_A2_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029211, upper bound: 0.0030560
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0030807
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0031177
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031321
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031455
IS_A2_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028431, upper bound: 0.0030336
IS_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028431, upper bound: 0.0030722
IS_A2_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031211
IS_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029421, upper bound: 0.0031344
IS_A2_B1_B2_A1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028236, upper bound: 0.0028503
IS_A2_B1_B2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027933, upper bound: 0.0028464
IS_A2_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028688, upper bound: 0.0029200
IS_A2_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028688, upper bound: 0.0030575
IS_A2_B1_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028582, upper bound: 0.0030059
IS_A2_B1_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028370, upper bound: 0.0030081
IS_A2_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0030574
IS_A2_B1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0028708, upper bound: 0.0031439
IS_A2_B1_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027899, upper bound: 0.0028483
IS_A2_B1_B2_A2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0026595, upper bound: 0.0027526
IS_A2_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029047, upper bound: 0.0028325
IS_A2_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029047, upper bound: 0.0030545
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030037
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030484
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0031211
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029169, upper bound: 0.0031344
IS_A2_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029945, upper bound: 0.0029321
IS_A2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029876, upper bound: 0.0029242
IS_A2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030623, upper bound: 0.0030147
IS_A2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030623, upper bound: 0.0030816
IS_A2_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030201, upper bound: 0.0030782
IS_A2_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030127, upper bound: 0.0030793
IS_A2_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030789, upper bound: 0.0031310
IS_A2_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030789, upper bound: 0.0031545
IS_A2_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029424, upper bound: 0.0027979
IS_A2_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029308, upper bound: 0.0027466
IS_A2_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030479, upper bound: 0.0029783
IS_A2_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030479, upper bound: 0.0030770
IS_A2_B2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029716, upper bound: 0.0029920
IS_A2_B2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029625, upper bound: 0.0029833
IS_A2_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030624, upper bound: 0.0030849
IS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030624, upper bound: 0.0031484
IS_A2_B2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027956, upper bound: 0.0029982
IS_A2_B2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0027466, upper bound: 0.0029584
IS_A2_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029257, upper bound: 0.0030904
IS_A2_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029257, upper bound: 0.0031549
IS_A2_B2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029668, upper bound: 0.0030297
IS_A2_B2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029551, upper bound: 0.0030001
IS_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030189, upper bound: 0.0031082
IS_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030189, upper bound: 0.0031615
IS_A2_B2_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029287, upper bound: 0.0027978
IS_A2_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029170, upper bound: 0.0027466
IS_A2_B2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030017, upper bound: 0.0029476
IS_A2_B2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030017, upper bound: 0.0030764
IS_A2_B2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029602, upper bound: 0.0029920
IS_A2_B2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0029507, upper bound: 0.0029833
IS_A2_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030246, upper bound: 0.0030631
IS_A2_B2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 1, lower bound: -0.0030246, upper bound: 0.0031484

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002039, 0.0019524, -0.0002533, 0.0020477, -0.0022516, 0.0022057
1: 0.9917448, 0.9963111, 0.9915429, 0.9964155, -0.0046707, 0.0047682
2: -0.0081079, -0.0045322, -0.0081326, -0.0042030, -0.0039049, 0.0036004
3: 0.0024330, 0.0051306, 0.0023712, 0.0052499, -0.0028169, 0.0027594
4: 0.0019990, 0.0062829, 0.0017388, 0.0066114, -0.0046124, 0.0045441
5: 0.0028791, 0.0079888, 0.0027621, 0.0082147, -0.0053357, 0.0052267
6: -0.0037304, 0.0011648, -0.0039886, 0.0012729, -0.0050033, 0.0051534
7: -0.0087193, -0.0065336, -0.0088159, -0.0064836, -0.0022357, 0.0022824
8: 0.0042064, 0.0082178, 0.0037738, 0.0082244, -0.0040027, 0.0044110
9: -0.0048864, -0.0017652, -0.0050244, -0.0016938, -0.0031926, 0.0032591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028748, upper bound: 0.0027704
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028748, upper bound: 0.0027993
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0001895, 0.0020128, -0.0002463, 0.0020467, -0.0022361, 0.0022590
1: 0.9916168, 0.9962804, 0.9915451, 0.9964008, -0.0047839, 0.0047354
2: -0.0081006, -0.0043237, -0.0081291, -0.0042065, -0.0038942, 0.0038054
3: 0.0024511, 0.0052062, 0.0023800, 0.0052486, -0.0027976, 0.0028262
4: 0.0018342, 0.0064910, 0.0017416, 0.0066079, -0.0047737, 0.0047494
5: 0.0029134, 0.0081319, 0.0027787, 0.0082123, -0.0052989, 0.0053532
6: -0.0038940, 0.0011330, -0.0039859, 0.0012576, -0.0051516, 0.0051189
7: -0.0087805, -0.0065483, -0.0088149, -0.0064907, -0.0022899, 0.0022666
8: 0.0039324, 0.0082159, 0.0037784, 0.0082235, -0.0042747, 0.0044288
9: -0.0049738, -0.0017862, -0.0050229, -0.0017039, -0.0032699, 0.0032367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028495, upper bound: 0.0027149
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028495, upper bound: 0.0027149
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001676, 0.0020273, -0.0003183, 0.0019577, -0.0021253, 0.0023456
1: 0.9915861, 0.9962342, 0.9917334, 0.9965532, -0.0049671, 0.0045008
2: -0.0080897, -0.0042735, -0.0081651, -0.0045138, -0.0035759, 0.0038917
3: 0.0024784, 0.0052244, 0.0022898, 0.0051373, -0.0026590, 0.0029345
4: 0.0017945, 0.0065411, 0.0019826, 0.0063013, -0.0045068, 0.0045585
5: 0.0029651, 0.0081664, 0.0026081, 0.0080015, -0.0050364, 0.0055583
6: -0.0039333, 0.0010853, -0.0037449, 0.0014154, -0.0053487, 0.0048301
7: -0.0087953, -0.0065704, -0.0087247, -0.0064177, -0.0023776, 0.0021543
8: 0.0038664, 0.0082130, 0.0041822, 0.0082331, -0.0043318, 0.0040156
9: -0.0049948, -0.0018178, -0.0048941, -0.0015997, -0.0033951, 0.0030763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029314, upper bound: 0.0028009
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028908, upper bound: 0.0027835
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001676, 0.0020273, -0.0002531, 0.0020312, -0.0021988, 0.0022804
1: 0.9915861, 0.9962342, 0.9915779, 0.9964153, -0.0048292, 0.0046564
2: -0.0080897, -0.0042735, -0.0081325, -0.0042600, -0.0038298, 0.0038590
3: 0.0024784, 0.0052244, 0.0023714, 0.0052293, -0.0027509, 0.0028530
4: 0.0017945, 0.0065411, 0.0017839, 0.0065546, -0.0047600, 0.0047572
5: 0.0029651, 0.0081664, 0.0027625, 0.0081756, -0.0052105, 0.0054038
6: -0.0039333, 0.0010853, -0.0039439, 0.0012726, -0.0052059, 0.0050292
7: -0.0087953, -0.0065704, -0.0087992, -0.0064838, -0.0023115, 0.0022288
8: 0.0038664, 0.0082130, 0.0038487, 0.0082244, -0.0043340, 0.0043448
9: -0.0049948, -0.0018178, -0.0050005, -0.0016941, -0.0033008, 0.0031827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029314, upper bound: 0.0028311
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028908, upper bound: 0.0028311
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002755, 0.0019510, -0.0002667, 0.0020483, -0.0023239, 0.0022177
1: 0.9917476, 0.9964626, 0.9915415, 0.9964439, -0.0046963, 0.0049211
2: -0.0081437, -0.0045368, -0.0081393, -0.0042008, -0.0039429, 0.0036024
3: 0.0023434, 0.0051290, 0.0023545, 0.0052507, -0.0029073, 0.0027745
4: 0.0020027, 0.0062783, 0.0017371, 0.0066136, -0.0046109, 0.0045412
5: 0.0027094, 0.0079857, 0.0027304, 0.0082162, -0.0055068, 0.0052553
6: -0.0037268, 0.0013217, -0.0039903, 0.0013023, -0.0050291, 0.0053120
7: -0.0087180, -0.0064610, -0.0088166, -0.0064700, -0.0022480, 0.0023556
8: 0.0042125, 0.0082274, 0.0037709, 0.0082262, -0.0039990, 0.0044167
9: -0.0048845, -0.0016616, -0.0050253, -0.0016744, -0.0032100, 0.0033637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029002, upper bound: 0.0029215
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029002, upper bound: 0.0029546
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002588, 0.0020107, -0.0002596, 0.0020473, -0.0023061, 0.0022704
1: 0.9916211, 0.9964273, 0.9915437, 0.9964291, -0.0048079, 0.0048836
2: -0.0081354, -0.0043306, -0.0081358, -0.0042043, -0.0039311, 0.0038051
3: 0.0023643, 0.0052037, 0.0023633, 0.0052494, -0.0028851, 0.0028404
4: 0.0018397, 0.0064841, 0.0017398, 0.0066101, -0.0047704, 0.0047442
5: 0.0027490, 0.0081272, 0.0027470, 0.0082138, -0.0054648, 0.0053801
6: -0.0038885, 0.0012851, -0.0039876, 0.0012869, -0.0051754, 0.0052727
7: -0.0087785, -0.0064780, -0.0088156, -0.0064771, -0.0023014, 0.0023376
8: 0.0039415, 0.0082251, 0.0037755, 0.0082253, -0.0042672, 0.0044341
9: -0.0049709, -0.0016858, -0.0050238, -0.0016846, -0.0032863, 0.0033380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028904, upper bound: 0.0028904
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028904, upper bound: 0.0028904
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002399, 0.0020290, -0.0003316, 0.0019584, -0.0021983, 0.0023605
1: 0.9915825, 0.9963874, 0.9917320, 0.9965814, -0.0049989, 0.0046554
2: -0.0081259, -0.0042677, -0.0081717, -0.0045115, -0.0036144, 0.0039040
3: 0.0023879, 0.0052264, 0.0022733, 0.0051382, -0.0027503, 0.0029532
4: 0.0017900, 0.0065468, 0.0019610, 0.0063036, -0.0045136, 0.0045858
5: 0.0027937, 0.0081703, 0.0025766, 0.0080031, -0.0052093, 0.0055937
6: -0.0039378, 0.0012437, -0.0037467, 0.0014445, -0.0053823, 0.0049904
7: -0.0087969, -0.0064971, -0.0087254, -0.0064042, -0.0023927, 0.0022283
8: 0.0038589, 0.0082226, 0.0041792, 0.0082349, -0.0043405, 0.0040220
9: -0.0049972, -0.0017131, -0.0048951, -0.0015805, -0.0034167, 0.0031820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0029894
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0030268
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002399, 0.0020290, -0.0002665, 0.0020318, -0.0022718, 0.0022955
1: 0.9915825, 0.9963874, 0.9915764, 0.9964435, -0.0048610, 0.0048109
2: -0.0081259, -0.0042677, -0.0081392, -0.0042578, -0.0038682, 0.0038715
3: 0.0023879, 0.0052264, 0.0023547, 0.0052301, -0.0028422, 0.0028718
4: 0.0017900, 0.0065468, 0.0017821, 0.0065567, -0.0047667, 0.0047647
5: 0.0027937, 0.0081703, 0.0027308, 0.0081771, -0.0053834, 0.0054395
6: -0.0039378, 0.0012437, -0.0039456, 0.0013019, -0.0052397, 0.0051893
7: -0.0087969, -0.0064971, -0.0087999, -0.0064702, -0.0023268, 0.0023028
8: 0.0038589, 0.0082226, 0.0038458, 0.0082262, -0.0043428, 0.0043509
9: -0.0049972, -0.0017131, -0.0050014, -0.0016747, -0.0033226, 0.0032883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0030479
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028433, upper bound: 0.0030676
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003183, 0.0019577, -0.0002344, 0.0020296, -0.0023479, 0.0021921
1: 0.9917334, 0.9965532, 0.9915812, 0.9963756, -0.0046421, 0.0049720
2: -0.0081651, -0.0045138, -0.0081231, -0.0042655, -0.0038996, 0.0036093
3: 0.0022898, 0.0051373, 0.0023948, 0.0052273, -0.0029374, 0.0027425
4: 0.0019826, 0.0063013, 0.0017882, 0.0065491, -0.0045664, 0.0045131
5: 0.0026081, 0.0080015, 0.0028069, 0.0081718, -0.0055638, 0.0051946
6: -0.0037449, 0.0014154, -0.0039396, 0.0012315, -0.0049764, 0.0053550
7: -0.0087247, -0.0064177, -0.0087976, -0.0065027, -0.0022220, 0.0023799
8: 0.0041822, 0.0082331, 0.0038559, 0.0082219, -0.0040224, 0.0043491
9: -0.0048941, -0.0015997, -0.0049982, -0.0017212, -0.0031730, 0.0033985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023703, upper bound: 0.0029057
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023703, upper bound: 0.0028688
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002531, 0.0020312, -0.0002344, 0.0020296, -0.0022827, 0.0022656
1: 0.9915779, 0.9964153, 0.9915812, 0.9963756, -0.0047977, 0.0048341
2: -0.0081325, -0.0042600, -0.0081231, -0.0042655, -0.0038670, 0.0038632
3: 0.0023714, 0.0052293, 0.0023948, 0.0052273, -0.0028559, 0.0028344
4: 0.0017839, 0.0065546, 0.0017882, 0.0065491, -0.0047652, 0.0047664
5: 0.0027625, 0.0081756, 0.0028069, 0.0081718, -0.0054093, 0.0053687
6: -0.0039439, 0.0012726, -0.0039396, 0.0012315, -0.0051755, 0.0052122
7: -0.0087992, -0.0064838, -0.0087976, -0.0065027, -0.0022965, 0.0023138
8: 0.0038487, 0.0082244, 0.0038559, 0.0082219, -0.0043497, 0.0043483
9: -0.0050005, -0.0016941, -0.0049982, -0.0017212, -0.0032793, 0.0033041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023703, upper bound: 0.0029620
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023703, upper bound: 0.0029433
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002667, 0.0020483, -0.0003330, 0.0019390, -0.0022057, 0.0023813
1: 0.9915415, 0.9964439, 0.9917731, 0.9965844, -0.0050429, 0.0046709
2: -0.0081393, -0.0042008, -0.0081724, -0.0045784, -0.0035608, 0.0039717
3: 0.0023545, 0.0052507, 0.0022715, 0.0051139, -0.0027595, 0.0029792
4: 0.0017371, 0.0066136, 0.0019587, 0.0062368, -0.0044997, 0.0046550
5: 0.0027304, 0.0082162, 0.0025732, 0.0079571, -0.0052267, 0.0056430
6: -0.0039903, 0.0013023, -0.0036942, 0.0014476, -0.0054380, 0.0049965
7: -0.0088166, -0.0064700, -0.0087058, -0.0064028, -0.0024138, 0.0022358
8: 0.0037709, 0.0082262, 0.0042671, 0.0082350, -0.0044206, 0.0039479
9: -0.0050253, -0.0016744, -0.0048670, -0.0015784, -0.0034469, 0.0031926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027694, upper bound: 0.0028295
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027694, upper bound: 0.0028548
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003316, 0.0019584, -0.0003130, 0.0020349, -0.0023664, 0.0022714
1: 0.9917320, 0.9965814, 0.9915700, 0.9965421, -0.0048101, 0.0050114
2: -0.0081717, -0.0045115, -0.0081625, -0.0042473, -0.0039244, 0.0036510
3: 0.0022733, 0.0051382, 0.0022965, 0.0052338, -0.0029606, 0.0028417
4: 0.0019610, 0.0063036, 0.0017739, 0.0065672, -0.0046062, 0.0045297
5: 0.0025766, 0.0080031, 0.0026206, 0.0081843, -0.0056077, 0.0053825
6: -0.0037467, 0.0014445, -0.0039538, 0.0014038, -0.0051505, 0.0053983
7: -0.0087254, -0.0064042, -0.0088029, -0.0064230, -0.0023024, 0.0023987
8: 0.0041792, 0.0082349, 0.0038321, 0.0082324, -0.0040295, 0.0043758
9: -0.0048951, -0.0015805, -0.0050058, -0.0016073, -0.0032878, 0.0034253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0028431
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030167
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002665, 0.0020318, -0.0003130, 0.0020349, -0.0023014, 0.0023448
1: 0.9915764, 0.9964435, 0.9915700, 0.9965421, -0.0049657, 0.0048735
2: -0.0081392, -0.0042578, -0.0081625, -0.0042473, -0.0038919, 0.0039047
3: 0.0023547, 0.0052301, 0.0022965, 0.0052338, -0.0028792, 0.0029336
4: 0.0017821, 0.0065567, 0.0017739, 0.0065672, -0.0047850, 0.0047829
5: 0.0027308, 0.0081771, 0.0026206, 0.0081843, -0.0054535, 0.0055565
6: -0.0039456, 0.0013019, -0.0039538, 0.0014038, -0.0053495, 0.0052557
7: -0.0087999, -0.0064702, -0.0088029, -0.0064230, -0.0023768, 0.0023328
8: 0.0038458, 0.0082262, 0.0038321, 0.0082324, -0.0043564, 0.0043748
9: -0.0050014, -0.0016747, -0.0050058, -0.0016073, -0.0033941, 0.0033311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0029421
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030556
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002344, 0.0020296, -0.0003183, 0.0019577, -0.0021921, 0.0023479
1: 0.9915812, 0.9963756, 0.9917334, 0.9965532, -0.0049720, 0.0046421
2: -0.0081231, -0.0042655, -0.0081651, -0.0045138, -0.0036093, 0.0038996
3: 0.0023948, 0.0052273, 0.0022898, 0.0051373, -0.0027425, 0.0029374
4: 0.0017882, 0.0065491, 0.0019826, 0.0063013, -0.0045131, 0.0045664
5: 0.0028069, 0.0081718, 0.0026081, 0.0080015, -0.0051946, 0.0055638
6: -0.0039396, 0.0012315, -0.0037449, 0.0014154, -0.0053550, 0.0049764
7: -0.0087976, -0.0065027, -0.0087247, -0.0064177, -0.0023799, 0.0022220
8: 0.0038559, 0.0082219, 0.0041822, 0.0082331, -0.0043491, 0.0040224
9: -0.0049982, -0.0017212, -0.0048941, -0.0015997, -0.0033985, 0.0031730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029057, upper bound: 0.0027497
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028688, upper bound: 0.0027297
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002344, 0.0020296, -0.0002531, 0.0020312, -0.0022656, 0.0022827
1: 0.9915812, 0.9963756, 0.9915779, 0.9964153, -0.0048341, 0.0047977
2: -0.0081231, -0.0042655, -0.0081325, -0.0042600, -0.0038632, 0.0038670
3: 0.0023948, 0.0052273, 0.0023714, 0.0052293, -0.0028344, 0.0028559
4: 0.0017882, 0.0065491, 0.0017839, 0.0065546, -0.0047664, 0.0047652
5: 0.0028069, 0.0081718, 0.0027625, 0.0081756, -0.0053687, 0.0054093
6: -0.0039396, 0.0012315, -0.0039439, 0.0012726, -0.0052122, 0.0051755
7: -0.0087976, -0.0065027, -0.0087992, -0.0064838, -0.0023138, 0.0022965
8: 0.0038559, 0.0082219, 0.0038487, 0.0082244, -0.0043483, 0.0043497
9: -0.0049982, -0.0017212, -0.0050005, -0.0016941, -0.0033041, 0.0032793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029057, upper bound: 0.0027497
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028688, upper bound: 0.0028057
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0003330, 0.0019390, -0.0002667, 0.0020483, -0.0023813, 0.0022057
1: 0.9917731, 0.9965844, 0.9915415, 0.9964439, -0.0046709, 0.0050429
2: -0.0081724, -0.0045784, -0.0081393, -0.0042008, -0.0039717, 0.0035608
3: 0.0022715, 0.0051139, 0.0023545, 0.0052507, -0.0029792, 0.0027595
4: 0.0019587, 0.0062368, 0.0017371, 0.0066136, -0.0046550, 0.0044997
5: 0.0025732, 0.0079571, 0.0027304, 0.0082162, -0.0056430, 0.0052267
6: -0.0036942, 0.0014476, -0.0039903, 0.0013023, -0.0049965, 0.0054380
7: -0.0087058, -0.0064028, -0.0088166, -0.0064700, -0.0022358, 0.0024138
8: 0.0042671, 0.0082350, 0.0037709, 0.0082262, -0.0039479, 0.0044206
9: -0.0048670, -0.0015784, -0.0050253, -0.0016744, -0.0031926, 0.0034469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028295, upper bound: 0.0027694
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028295, upper bound: 0.0028088
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003130, 0.0020349, -0.0003316, 0.0019584, -0.0022714, 0.0023664
1: 0.9915700, 0.9965421, 0.9917320, 0.9965814, -0.0050114, 0.0048101
2: -0.0081625, -0.0042473, -0.0081717, -0.0045115, -0.0036510, 0.0039244
3: 0.0022965, 0.0052338, 0.0022733, 0.0051382, -0.0028417, 0.0029606
4: 0.0017739, 0.0065672, 0.0019610, 0.0063036, -0.0045297, 0.0046062
5: 0.0026206, 0.0081843, 0.0025766, 0.0080031, -0.0053825, 0.0056077
6: -0.0039538, 0.0014038, -0.0037467, 0.0014445, -0.0053983, 0.0051505
7: -0.0088029, -0.0064230, -0.0087254, -0.0064042, -0.0023987, 0.0023024
8: 0.0038321, 0.0082324, 0.0041792, 0.0082349, -0.0043758, 0.0040295
9: -0.0050058, -0.0016073, -0.0048951, -0.0015805, -0.0034253, 0.0032878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028430, upper bound: 0.0029266
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028430, upper bound: 0.0029608
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003130, 0.0020349, -0.0002665, 0.0020318, -0.0023448, 0.0023014
1: 0.9915700, 0.9965421, 0.9915764, 0.9964435, -0.0048735, 0.0049657
2: -0.0081625, -0.0042473, -0.0081392, -0.0042578, -0.0039047, 0.0038919
3: 0.0022965, 0.0052338, 0.0023547, 0.0052301, -0.0029336, 0.0028792
4: 0.0017739, 0.0065672, 0.0017821, 0.0065567, -0.0047829, 0.0047850
5: 0.0026206, 0.0081843, 0.0027308, 0.0081771, -0.0055565, 0.0054535
6: -0.0039538, 0.0014038, -0.0039456, 0.0013019, -0.0052557, 0.0053495
7: -0.0088029, -0.0064230, -0.0087999, -0.0064702, -0.0023328, 0.0023768
8: 0.0038321, 0.0082324, 0.0038458, 0.0082262, -0.0043748, 0.0043564
9: -0.0050058, -0.0016073, -0.0050014, -0.0016747, -0.0033311, 0.0033941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028430, upper bound: 0.0030265
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028430, upper bound: 0.0030436
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002344, 0.0020296, -0.0003721, 0.0019461, -0.0021805, 0.0024018
1: 0.9915812, 0.9963756, 0.9917580, 0.9966673, -0.0050861, 0.0046175
2: -0.0081231, -0.0042655, -0.0081920, -0.0045540, -0.0035691, 0.0039266
3: 0.0023948, 0.0052273, 0.0022225, 0.0051228, -0.0027279, 0.0030048
4: 0.0017882, 0.0065491, 0.0018948, 0.0062612, -0.0044730, 0.0046543
5: 0.0028069, 0.0081718, 0.0024804, 0.0079739, -0.0051670, 0.0056914
6: -0.0039396, 0.0012315, -0.0037134, 0.0015334, -0.0054730, 0.0049449
7: -0.0087976, -0.0065027, -0.0087129, -0.0063631, -0.0024345, 0.0022102
8: 0.0038559, 0.0082219, 0.0042350, 0.0082403, -0.0043450, 0.0039672
9: -0.0049982, -0.0017212, -0.0048773, -0.0015218, -0.0034764, 0.0031561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027785, upper bound: 0.0026994
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026595, upper bound: 0.0026459
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002344, 0.0020296, -0.0003234, 0.0020373, -0.0022717, 0.0023530
1: 0.9915812, 0.9963756, 0.9915649, 0.9965641, -0.0049829, 0.0048106
2: -0.0081231, -0.0042655, -0.0081676, -0.0042389, -0.0038843, 0.0039022
3: 0.0023948, 0.0052273, 0.0022835, 0.0052369, -0.0028421, 0.0029438
4: 0.0017882, 0.0065491, 0.0017672, 0.0065756, -0.0047874, 0.0047819
5: 0.0028069, 0.0081718, 0.0025960, 0.0081901, -0.0053832, 0.0055758
6: -0.0039396, 0.0012315, -0.0039605, 0.0014266, -0.0053662, 0.0051920
7: -0.0087976, -0.0065027, -0.0088054, -0.0064125, -0.0023851, 0.0023027
8: 0.0038559, 0.0082219, 0.0038210, 0.0082338, -0.0043511, 0.0043790
9: -0.0049982, -0.0017212, -0.0050093, -0.0015923, -0.0034059, 0.0032882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027785, upper bound: 0.0026994
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0026595, upper bound: 0.0028057
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003130, 0.0020349, -0.0003858, 0.0019467, -0.0022597, 0.0024206
1: 0.9915700, 0.9965421, 0.9917567, 0.9966961, -0.0051261, 0.0047854
2: -0.0081625, -0.0042473, -0.0081989, -0.0045518, -0.0036107, 0.0039515
3: 0.0022965, 0.0052338, 0.0022055, 0.0051236, -0.0028271, 0.0030284
4: 0.0017739, 0.0065672, 0.0018726, 0.0062634, -0.0044895, 0.0046946
5: 0.0026206, 0.0081843, 0.0024481, 0.0079754, -0.0053548, 0.0057362
6: -0.0039538, 0.0014038, -0.0037151, 0.0015633, -0.0055171, 0.0051189
7: -0.0088029, -0.0064230, -0.0087136, -0.0063493, -0.0024537, 0.0022905
8: 0.0038321, 0.0082324, 0.0042321, 0.0082421, -0.0043710, 0.0039738
9: -0.0050058, -0.0016073, -0.0048782, -0.0015020, -0.0035038, 0.0032709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0029047
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0029410
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003130, 0.0020349, -0.0003372, 0.0020379, -0.0023509, 0.0023721
1: 0.9915700, 0.9965421, 0.9915636, 0.9965933, -0.0050233, 0.0049785
2: -0.0081625, -0.0042473, -0.0081746, -0.0042367, -0.0039257, 0.0039273
3: 0.0022965, 0.0052338, 0.0022662, 0.0052377, -0.0029412, 0.0029677
4: 0.0017739, 0.0065672, 0.0017655, 0.0065777, -0.0048039, 0.0048017
5: 0.0026206, 0.0081843, 0.0025632, 0.0081916, -0.0055710, 0.0056211
6: -0.0039538, 0.0014038, -0.0039621, 0.0014569, -0.0054107, 0.0053660
7: -0.0088029, -0.0064230, -0.0088060, -0.0063985, -0.0024044, 0.0023830
8: 0.0038321, 0.0082324, 0.0038182, 0.0082356, -0.0043767, 0.0043853
9: -0.0050058, -0.0016073, -0.0050102, -0.0015723, -0.0034335, 0.0034029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030265
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0026852, upper bound: 0.0030429
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002897, 0.0019533, -0.0002908, 0.0020564, -0.0023461, 0.0022441
1: 0.9917429, 0.9964927, 0.9915244, 0.9964951, -0.0047522, 0.0049683
2: -0.0081508, -0.0045292, -0.0081514, -0.0041727, -0.0039780, 0.0036222
3: 0.0023257, 0.0051317, 0.0023243, 0.0052608, -0.0029352, 0.0028075
4: 0.0019967, 0.0062860, 0.0017149, 0.0066416, -0.0046449, 0.0045710
5: 0.0026759, 0.0079909, 0.0026732, 0.0082354, -0.0055596, 0.0053177
6: -0.0037328, 0.0013527, -0.0040123, 0.0013552, -0.0050880, 0.0053650
7: -0.0087202, -0.0064467, -0.0088248, -0.0064455, -0.0022747, 0.0023781
8: 0.0042024, 0.0082293, 0.0037341, 0.0082294, -0.0040102, 0.0044582
9: -0.0048877, -0.0016411, -0.0050370, -0.0016395, -0.0032482, 0.0033959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029514, upper bound: 0.0028025
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029514, upper bound: 0.0029315
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002731, 0.0020130, -0.0002844, 0.0020555, -0.0023287, 0.0022973
1: 0.9916164, 0.9964577, 0.9915263, 0.9964815, -0.0048651, 0.0049314
2: -0.0081425, -0.0043230, -0.0081481, -0.0041759, -0.0039666, 0.0038252
3: 0.0023463, 0.0052064, 0.0023323, 0.0052597, -0.0029134, 0.0028741
4: 0.0018337, 0.0064917, 0.0017174, 0.0066384, -0.0048048, 0.0047743
5: 0.0027150, 0.0081324, 0.0026884, 0.0082333, -0.0055183, 0.0054439
6: -0.0038945, 0.0013165, -0.0040098, 0.0013411, -0.0052356, 0.0053263
7: -0.0087807, -0.0064634, -0.0088239, -0.0064520, -0.0023287, 0.0023605
8: 0.0039315, 0.0082271, 0.0037382, 0.0082286, -0.0042787, 0.0044761
9: -0.0049741, -0.0016650, -0.0050357, -0.0016488, -0.0033253, 0.0033707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028968, upper bound: 0.0027149
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028968, upper bound: 0.0028912
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002531, 0.0020312, -0.0003612, 0.0019725, -0.0022256, 0.0023924
1: 0.9915779, 0.9964153, 0.9917022, 0.9966441, -0.0050662, 0.0047131
2: -0.0081325, -0.0042600, -0.0081866, -0.0044628, -0.0036697, 0.0039266
3: 0.0023714, 0.0052293, 0.0022362, 0.0051558, -0.0027844, 0.0029931
4: 0.0017839, 0.0065546, 0.0019127, 0.0063521, -0.0045683, 0.0046419
5: 0.0027625, 0.0081756, 0.0025064, 0.0080364, -0.0052739, 0.0056692
6: -0.0039439, 0.0012726, -0.0037849, 0.0015094, -0.0054533, 0.0050574
7: -0.0087992, -0.0064838, -0.0087397, -0.0063742, -0.0024250, 0.0022559
8: 0.0038487, 0.0082244, 0.0041152, 0.0082388, -0.0043561, 0.0040910
9: -0.0050005, -0.0016941, -0.0049155, -0.0015376, -0.0034629, 0.0032214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028918, upper bound: 0.0028973
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028875, upper bound: 0.0028695
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002531, 0.0020312, -0.0002906, 0.0020367, -0.0022898, 0.0023218
1: 0.9915779, 0.9964153, 0.9915661, 0.9964945, -0.0049166, 0.0048492
2: -0.0081325, -0.0042600, -0.0081512, -0.0042409, -0.0038916, 0.0038913
3: 0.0023714, 0.0052293, 0.0023246, 0.0052362, -0.0028648, 0.0029047
4: 0.0017839, 0.0065546, 0.0017688, 0.0065736, -0.0047897, 0.0047858
5: 0.0027625, 0.0081756, 0.0026738, 0.0081887, -0.0054262, 0.0055018
6: -0.0039439, 0.0012726, -0.0039589, 0.0013546, -0.0052985, 0.0052314
7: -0.0087992, -0.0064838, -0.0088048, -0.0064458, -0.0023534, 0.0023210
8: 0.0038487, 0.0082244, 0.0038237, 0.0082294, -0.0043561, 0.0043782
9: -0.0050005, -0.0016941, -0.0050085, -0.0016399, -0.0033606, 0.0033144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028918, upper bound: 0.0029647
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028875, upper bound: 0.0029549
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002897, 0.0019533, -0.0003477, 0.0020480, -0.0023377, 0.0023010
1: 0.9917429, 0.9964927, 0.9915422, 0.9966156, -0.0048727, 0.0049505
2: -0.0081508, -0.0045292, -0.0081798, -0.0042020, -0.0039488, 0.0036506
3: 0.0023257, 0.0051317, 0.0022531, 0.0052503, -0.0029246, 0.0028786
4: 0.0019967, 0.0062860, 0.0017380, 0.0066124, -0.0046158, 0.0045479
5: 0.0026759, 0.0079909, 0.0025384, 0.0082154, -0.0055395, 0.0054525
6: -0.0037328, 0.0013527, -0.0039894, 0.0014798, -0.0052126, 0.0053421
7: -0.0087202, -0.0064467, -0.0088162, -0.0063879, -0.0023323, 0.0023696
8: 0.0042024, 0.0082293, 0.0037725, 0.0082370, -0.0040151, 0.0044224
9: -0.0048877, -0.0016411, -0.0050248, -0.0015572, -0.0033305, 0.0033837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027476, upper bound: 0.0027805
time: 1.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027476, upper bound: 0.0029065
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002731, 0.0020130, -0.0003408, 0.0020471, -0.0023202, 0.0023538
1: 0.9916164, 0.9964577, 0.9915441, 0.9966009, -0.0049845, 0.0049136
2: -0.0081425, -0.0043230, -0.0081764, -0.0042051, -0.0039374, 0.0038534
3: 0.0023463, 0.0052064, 0.0022617, 0.0052491, -0.0029028, 0.0029447
4: 0.0018337, 0.0064917, 0.0017405, 0.0066093, -0.0047756, 0.0047512
5: 0.0027150, 0.0081324, 0.0025547, 0.0082132, -0.0054982, 0.0055777
6: -0.0038945, 0.0013165, -0.0039869, 0.0014647, -0.0053592, 0.0053034
7: -0.0087807, -0.0064634, -0.0088153, -0.0063949, -0.0023859, 0.0023519
8: 0.0039315, 0.0082271, 0.0037766, 0.0082361, -0.0042835, 0.0044403
9: -0.0049741, -0.0016650, -0.0050235, -0.0015671, -0.0034070, 0.0033584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028431, upper bound: 0.0027149
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028431, upper bound: 0.0028715
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002531, 0.0020312, -0.0004099, 0.0019551, -0.0022082, 0.0024411
1: 0.9915779, 0.9964153, 0.9917390, 0.9967474, -0.0051695, 0.0046763
2: -0.0081325, -0.0042600, -0.0082109, -0.0045228, -0.0036097, 0.0039510
3: 0.0023714, 0.0052293, 0.0021752, 0.0051341, -0.0027627, 0.0030540
4: 0.0017839, 0.0065546, 0.0018332, 0.0062923, -0.0045085, 0.0047213
5: 0.0027625, 0.0081756, 0.0023910, 0.0079953, -0.0052328, 0.0057847
6: -0.0039439, 0.0012726, -0.0037378, 0.0016162, -0.0055601, 0.0050104
7: -0.0087992, -0.0064838, -0.0087221, -0.0063248, -0.0024744, 0.0022383
8: 0.0038487, 0.0082244, 0.0041940, 0.0082453, -0.0043585, 0.0040146
9: -0.0050005, -0.0016941, -0.0048904, -0.0014671, -0.0035334, 0.0031963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027500, upper bound: 0.0028045
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027238, upper bound: 0.0027539
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002531, 0.0020312, -0.0003476, 0.0020330, -0.0022861, 0.0023788
1: 0.9915779, 0.9964153, 0.9915740, 0.9966153, -0.0050374, 0.0048413
2: -0.0081325, -0.0042600, -0.0081797, -0.0042538, -0.0038787, 0.0039198
3: 0.0023714, 0.0052293, 0.0022532, 0.0052315, -0.0028601, 0.0029760
4: 0.0017839, 0.0065546, 0.0017790, 0.0065607, -0.0047768, 0.0047756
5: 0.0027625, 0.0081756, 0.0025386, 0.0081798, -0.0054173, 0.0056370
6: -0.0039439, 0.0012726, -0.0039487, 0.0014796, -0.0054235, 0.0052213
7: -0.0087992, -0.0064838, -0.0088010, -0.0063880, -0.0024112, 0.0023173
8: 0.0038487, 0.0082244, 0.0038406, 0.0082370, -0.0043593, 0.0043638
9: -0.0050005, -0.0016941, -0.0050031, -0.0015573, -0.0034432, 0.0033090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.23 + 597.71 = 600.94 seconds
