## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0051876


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004237, 0.0004237)
1: (0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023460, 0.0023460)
2: (0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052412, 0.0052412)
3: (0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022087, 0.0022087)
4: (1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085687, 0.0085687)
5: (0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016670, 0.0016670)
6: (-0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021693, 0.0021693)
7: (-0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002767, 0.0002767)
8: (-0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014988, 0.0014988)
9: (-0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075034, 0.0075034)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.64 = 3.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0062137, upper bound: 0.0062138

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0061148, upper bound: 0.0061833
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0061832, upper bound: 0.0061148
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 4, lower bound: -0.0061148, upper bound: 0.0061833
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 4, lower bound: -0.0061832, upper bound: 0.0061148

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004236, 0.0004228
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023411, 0.0023454
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052400, 0.0052302
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022040, 0.0022081
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085508, 0.0085667
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016635, 0.0016666
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021688, 0.0021648
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002767, 0.0002761
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014957, 0.0014985
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075017, 0.0074877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0061121, upper bound: 0.0060049
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059448, upper bound: 0.0061806
time: 0.85 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004228, 0.0004236
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023454, 0.0023411
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052302, 0.0052400
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022081, 0.0022040
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085667, 0.0085508
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016666, 0.0016635
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021648, 0.0021688
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002761, 0.0002767
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014985, 0.0014957
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074877, 0.0075017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0061533, upper bound: 0.0060073
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0060918, upper bound: 0.0060839
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 4, lower bound: -0.0061121, upper bound: 0.0060049
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 4, lower bound: -0.0059448, upper bound: 0.0061806
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 4, lower bound: -0.0061533, upper bound: 0.0060073
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 4, lower bound: -0.0060918, upper bound: 0.0060839

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004268, 0.0004295
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023783, 0.0023634
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052802, 0.0053134
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022391, 0.0022251
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086867, 0.0086325
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016899, 0.0016794
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021854, 0.0021992
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002788, 0.0002805
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015195, 0.0015100
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075593, 0.0076068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0060719, upper bound: 0.0058209
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058746, upper bound: 0.0059654
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004303, 0.0004261
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023591, 0.0023826
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053231, 0.0052704
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022210, 0.0022432
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086165, 0.0087026
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016763, 0.0016930
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022032, 0.0021814
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002810, 0.0002783
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015072, 0.0015222
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076207, 0.0075453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059055, upper bound: 0.0059268
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057567, upper bound: 0.0061400
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004236, 0.0004260
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023586, 0.0023454
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052399, 0.0052693
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022205, 0.0022081
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086147, 0.0085667
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016759, 0.0016666
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021688, 0.0021810
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002766, 0.0002782
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015069, 0.0014984
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075016, 0.0075437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057321, upper bound: 0.0054107
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055311, upper bound: 0.0055818
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004252, 0.0004244
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023498, 0.0023542
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052595, 0.0052497
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022122, 0.0022164
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085826, 0.0085987
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016697, 0.0016728
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021769, 0.0021728
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002777, 0.0002772
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015012, 0.0015041
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075297, 0.0075156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059997, upper bound: 0.0059344
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059336, upper bound: 0.0059917
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0060719, upper bound: 0.0058209
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0058746, upper bound: 0.0059654
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0059055, upper bound: 0.0059268
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0057567, upper bound: 0.0061400
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0057321, upper bound: 0.0054107
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0055311, upper bound: 0.0055818
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0059997, upper bound: 0.0059344
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -0.0059336, upper bound: 0.0059917

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004268, 0.0004296
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023788, 0.0023633
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052799, 0.0053146
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022396, 0.0022250
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086887, 0.0086320
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016903, 0.0016793
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021853, 0.0021997
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002788, 0.0002806
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015198, 0.0015099
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075588, 0.0076084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059427, upper bound: 0.0057174
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059652, upper bound: 0.0056932
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004269, 0.0004295
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023782, 0.0023640
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052814, 0.0053131
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022389, 0.0022256
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086862, 0.0086345
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016898, 0.0016798
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021860, 0.0021990
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002788, 0.0002805
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015194, 0.0015103
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075610, 0.0076063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056135, upper bound: 0.0052820
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052351, upper bound: 0.0057075
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004303, 0.0004262
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023596, 0.0023825
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053228, 0.0052716
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022215, 0.0022430
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086185, 0.0087021
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016766, 0.0016929
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022031, 0.0021819
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002810, 0.0002783
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015075, 0.0015221
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076202, 0.0075470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058757, upper bound: 0.0058694
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058453, upper bound: 0.0058974
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004304, 0.0004260
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023589, 0.0023832
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053243, 0.0052701
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022208, 0.0022437
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086160, 0.0087045
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016762, 0.0016934
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022037, 0.0021813
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002811, 0.0002782
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015071, 0.0015226
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076223, 0.0075448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053573, upper bound: 0.0055174
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052126, upper bound: 0.0057167
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004086, 0.0004299
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023805, 0.0022623
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050542, 0.0053183
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022412, 0.0021298
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086948, 0.0082630
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016915, 0.0016075
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020919, 0.0022012
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002668, 0.0002808
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015209, 0.0014453
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072357, 0.0076138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057286, upper bound: 0.0052937
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055334, upper bound: 0.0054080
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004236, 0.0004110
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022754, 0.0023454
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052399, 0.0050836
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021422, 0.0022081
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083111, 0.0085667
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016168, 0.0016666
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021688, 0.0021041
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002766, 0.0002684
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014537, 0.0014984
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075016, 0.0072778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054364, upper bound: 0.0054391
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054164, upper bound: 0.0054906
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004213, 0.0004230
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023423, 0.0023325
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052111, 0.0052329
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022052, 0.0021960
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085552, 0.0085196
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016643, 0.0016574
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021569, 0.0021659
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002751, 0.0002763
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014964, 0.0014902
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074604, 0.0074915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059970, upper bound: 0.0058047
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058660, upper bound: 0.0059317
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004239, 0.0004205
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023281, 0.0023469
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052433, 0.0052013
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021918, 0.0022095
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085034, 0.0085722
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016543, 0.0016676
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021702, 0.0021528
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002768, 0.0002746
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014874, 0.0014994
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075064, 0.0074462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059309, upper bound: 0.0058172
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058376, upper bound: 0.0059891
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0059427, upper bound: 0.0057174
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0059652, upper bound: 0.0056932
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0056135, upper bound: 0.0052820
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0052351, upper bound: 0.0057075
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0058757, upper bound: 0.0058694
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0058453, upper bound: 0.0058974
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0053573, upper bound: 0.0055174
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0052126, upper bound: 0.0057167
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0057286, upper bound: 0.0052937
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0055334, upper bound: 0.0054080
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0054364, upper bound: 0.0054391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0054164, upper bound: 0.0054906
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0059970, upper bound: 0.0058047
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0058660, upper bound: 0.0059317
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0059309, upper bound: 0.0058172
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0058376, upper bound: 0.0059891

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004084, 0.0004111
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022762, 0.0022611
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050515, 0.0050853
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021429, 0.0021287
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083138, 0.0082585
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016174, 0.0016066
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020908, 0.0021048
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002667, 0.0002685
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014542, 0.0014446
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072318, 0.0072802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053254, upper bound: 0.0051069
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052699, upper bound: 0.0051191
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004083, 0.0004112
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022770, 0.0022607
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050506, 0.0050871
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021437, 0.0021283
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083168, 0.0082571
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016179, 0.0016063
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020904, 0.0021055
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002667, 0.0002686
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014547, 0.0014443
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072305, 0.0072828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058749, upper bound: 0.0055822
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058191, upper bound: 0.0055949
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003793, 0.0003992
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022104, 0.0021001
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0046919, 0.0049382
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020810, 0.0019772
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0080734, 0.0076707
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015706, 0.0014922
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019419, 0.0020439
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002477, 0.0002607
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014122, 0.0013417
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0067170, 0.0070696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055207, upper bound: 0.0051472
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054652, upper bound: 0.0051889
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003964, 0.0003818
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021143, 0.0021948
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049034, 0.0047235
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019905, 0.0020663
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0077224, 0.0080165
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015023, 0.0015595
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020295, 0.0019550
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002589, 0.0002494
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013508, 0.0014022
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070198, 0.0067623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048348, upper bound: 0.0051158
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0046377, upper bound: 0.0052617
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004316, 0.0004290
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023756, 0.0023897
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053389, 0.0053074
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022365, 0.0022498
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086769, 0.0087284
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016880, 0.0016980
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022097, 0.0021967
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002819, 0.0002802
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015177, 0.0015267
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076433, 0.0075982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057784, upper bound: 0.0057130
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057658, upper bound: 0.0057795
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004332, 0.0004275
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023668, 0.0023987
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053590, 0.0052877
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022283, 0.0022583
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086448, 0.0087614
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016818, 0.0017044
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022181, 0.0021886
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002829, 0.0002792
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015121, 0.0015325
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076721, 0.0075701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057484, upper bound: 0.0057508
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057353, upper bound: 0.0058044
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004158, 0.0004299
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023805, 0.0023023
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051437, 0.0053184
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022412, 0.0021675
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086949, 0.0084093
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016915, 0.0016359
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021289, 0.0022013
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002716, 0.0002808
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015209, 0.0014709
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0073638, 0.0076139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050818, upper bound: 0.0048077
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0047071, upper bound: 0.0052766
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004304, 0.0004114
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022781, 0.0023832
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053243, 0.0050895
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021447, 0.0022437
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083208, 0.0087045
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016187, 0.0016934
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022037, 0.0021065
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002811, 0.0002687
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014554, 0.0015226
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076223, 0.0072863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041706, upper bound: 0.0043942
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041706, upper bound: 0.0043942
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004128, 0.0004381
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024255, 0.0022855
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051060, 0.0054189
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022835, 0.0021517
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088592, 0.0083476
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017235, 0.0016239
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021133, 0.0022428
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002696, 0.0002861
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015496, 0.0014601
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0073098, 0.0077578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056387, upper bound: 0.0051931
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055819, upper bound: 0.0051953
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004162, 0.0004341
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024037, 0.0023047
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051489, 0.0053701
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022630, 0.0021698
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0087795, 0.0084178
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017080, 0.0016376
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021311, 0.0022227
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002718, 0.0002835
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015357, 0.0014724
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0073713, 0.0076880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054116, upper bound: 0.0053122
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054386, upper bound: 0.0052252
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004197, 0.0004107
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022741, 0.0023237
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051915, 0.0050807
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021410, 0.0021877
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083063, 0.0084875
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016159, 0.0016512
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021487, 0.0021029
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002741, 0.0002682
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014529, 0.0014846
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074323, 0.0072736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053977, upper bound: 0.0052513
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052155, upper bound: 0.0054024
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004222, 0.0004081
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022598, 0.0023379
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052232, 0.0050487
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021275, 0.0022011
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0082540, 0.0085393
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016057, 0.0016612
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021619, 0.0020896
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002758, 0.0002666
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014438, 0.0014937
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074777, 0.0072278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052829, upper bound: 0.0053978
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053161, upper bound: 0.0053475
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004262, 0.0004317
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023901, 0.0023600
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052726, 0.0053397
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022502, 0.0022219
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0087298, 0.0086201
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016983, 0.0016770
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021823, 0.0022101
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002784, 0.0002819
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015270, 0.0015078
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075484, 0.0076445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059578, upper bound: 0.0056222
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057795, upper bound: 0.0057658
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004298, 0.0004280
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023698, 0.0023798
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053168, 0.0052944
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022311, 0.0022405
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086557, 0.0086923
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016839, 0.0016910
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022006, 0.0021913
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002807, 0.0002795
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015140, 0.0015204
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076116, 0.0075796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051985, upper bound: 0.0052537
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051832, upper bound: 0.0053094
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004288, 0.0004289
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023748, 0.0023745
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053048, 0.0053056
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022358, 0.0022354
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086741, 0.0086727
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016875, 0.0016872
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021956, 0.0021960
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002801, 0.0002801
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015172, 0.0015170
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075945, 0.0075957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057662, upper bound: 0.0057135
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058295, upper bound: 0.0056897
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004325, 0.0004254
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023556, 0.0023947
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053500, 0.0052627
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022177, 0.0022545
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086040, 0.0087466
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016738, 0.0017016
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022143, 0.0021782
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002825, 0.0002779
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015050, 0.0015299
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076592, 0.0075343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055734, upper bound: 0.0052707
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051350, upper bound: 0.0057394
time: 0.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0053254, upper bound: 0.0051069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0052699, upper bound: 0.0051191
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0058749, upper bound: 0.0055822
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0058191, upper bound: 0.0055949
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0055207, upper bound: 0.0051472
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0054652, upper bound: 0.0051889
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0048348, upper bound: 0.0051158
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0046377, upper bound: 0.0052617
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0057784, upper bound: 0.0057130
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0057658, upper bound: 0.0057795
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0057484, upper bound: 0.0057508
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0057353, upper bound: 0.0058044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0050818, upper bound: 0.0048077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0047071, upper bound: 0.0052766
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0041706, upper bound: 0.0043942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0041706, upper bound: 0.0043942
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0056387, upper bound: 0.0051931
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0055819, upper bound: 0.0051953
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0054116, upper bound: 0.0053122
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0054386, upper bound: 0.0052252
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0053977, upper bound: 0.0052513
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0052155, upper bound: 0.0054024
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0052829, upper bound: 0.0053978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0053161, upper bound: 0.0053475
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0059578, upper bound: 0.0056222
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0057795, upper bound: 0.0057658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0051985, upper bound: 0.0052537
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0051832, upper bound: 0.0053094
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0057662, upper bound: 0.0057135
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0058295, upper bound: 0.0056897
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0055734, upper bound: 0.0052707
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -0.0051350, upper bound: 0.0057394

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003971, 0.0004202
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023267, 0.0021985
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049116, 0.0051980
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021905, 0.0020698
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084981, 0.0080300
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016532, 0.0015621
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020329, 0.0021514
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002593, 0.0002744
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014865, 0.0014046
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070316, 0.0074416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053015, upper bound: 0.0050747
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052320, upper bound: 0.0050852
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004084, 0.0003998
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022136, 0.0022611
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050515, 0.0049454
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020840, 0.0021287
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0080852, 0.0082585
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015729, 0.0016066
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020908, 0.0020469
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002667, 0.0002611
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014142, 0.0014446
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072318, 0.0070800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0042029, upper bound: 0.0041238
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041959, upper bound: 0.0041241
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004062, 0.0004115
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022787, 0.0022492
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050249, 0.0050908
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021453, 0.0021175
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083228, 0.0082151
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016191, 0.0015982
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020798, 0.0021070
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002653, 0.0002688
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014558, 0.0014370
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0071938, 0.0072881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052368, upper bound: 0.0049434
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051837, upper bound: 0.0049563
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004088, 0.0004092
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022655, 0.0022633
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050566, 0.0050614
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021329, 0.0021308
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0082748, 0.0082669
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016098, 0.0016082
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020929, 0.0020949
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002670, 0.0002672
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014474, 0.0014460
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072391, 0.0072460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054106, upper bound: 0.0050198
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052613, upper bound: 0.0051873
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003744, 0.0003964
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021946, 0.0020731
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0046314, 0.0049030
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020661, 0.0019517
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0080158, 0.0075718
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015594, 0.0014730
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019169, 0.0020293
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002445, 0.0002589
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014021, 0.0013244
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0066305, 0.0070193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048905, upper bound: 0.0044826
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048704, upper bound: 0.0045055
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003770, 0.0003943
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021833, 0.0020872
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0046631, 0.0048777
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020555, 0.0019650
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0079745, 0.0076236
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015514, 0.0014831
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019300, 0.0020189
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002462, 0.0002575
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013949, 0.0013335
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0066758, 0.0069831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048374, upper bound: 0.0045026
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048063, upper bound: 0.0045209
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003964, 0.0003666
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0020299, 0.0021948
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049034, 0.0045351
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019111, 0.0020663
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0074144, 0.0080165
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014424, 0.0015595
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020295, 0.0018771
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002589, 0.0002394
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0012969, 0.0014022
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070198, 0.0064926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045127, upper bound: 0.0051678
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045420, upper bound: 0.0051308
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004289, 0.0004290
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023751, 0.0023748
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053056, 0.0053062
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022361, 0.0022358
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086751, 0.0086740
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016876, 0.0016874
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021959, 0.0021962
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002801, 0.0002801
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015174, 0.0015172
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075956, 0.0075965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051250, upper bound: 0.0050895
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050935, upper bound: 0.0051043
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004317, 0.0004264
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023607, 0.0023901
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053397, 0.0052741
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022225, 0.0022501
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086225, 0.0087297
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016774, 0.0016983
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022101, 0.0021829
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002819, 0.0002785
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015082, 0.0015270
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076444, 0.0075505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051168, upper bound: 0.0051632
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050851, upper bound: 0.0051858
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004305, 0.0004273
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023661, 0.0023838
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053257, 0.0052862
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022276, 0.0022443
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086423, 0.0087070
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016813, 0.0016938
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022043, 0.0021879
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002812, 0.0002791
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015117, 0.0015230
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076245, 0.0075678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056006, upper bound: 0.0056490
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056464, upper bound: 0.0056191
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004333, 0.0004248
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023519, 0.0023992
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053602, 0.0052545
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022142, 0.0022588
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085904, 0.0087632
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016712, 0.0017048
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022185, 0.0021748
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002830, 0.0002774
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015026, 0.0015328
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076737, 0.0075224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052990, upper bound: 0.0052128
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051565, upper bound: 0.0054030
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003847, 0.0003816
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021132, 0.0021300
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0047586, 0.0047211
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019895, 0.0020053
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0077184, 0.0077798
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015015, 0.0015135
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019696, 0.0019540
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002512, 0.0002493
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013501, 0.0013608
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0068126, 0.0067588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0046029, upper bound: 0.0051799
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0046088, upper bound: 0.0051404
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004110, 0.0004401
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024366, 0.0022758
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050845, 0.0054437
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022940, 0.0021426
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088999, 0.0083125
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017314, 0.0016171
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021044, 0.0022531
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002684, 0.0002874
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015567, 0.0014540
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072790, 0.0077934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055249, upper bound: 0.0050961
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055428, upper bound: 0.0050191
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004136, 0.0004363
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024159, 0.0022900
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051162, 0.0053974
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022745, 0.0021560
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088240, 0.0083643
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017166, 0.0016272
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021176, 0.0022339
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002701, 0.0002850
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015435, 0.0014631
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0073244, 0.0077270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055442, upper bound: 0.0050750
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053340, upper bound: 0.0051590
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003963, 0.0004140
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022924, 0.0021941
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049019, 0.0051214
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021582, 0.0020657
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083729, 0.0080141
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016289, 0.0015591
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020289, 0.0021197
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002588, 0.0002704
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014646, 0.0014018
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070177, 0.0073319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053162, upper bound: 0.0051947
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052949, upper bound: 0.0052196
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003961, 0.0004125
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022842, 0.0021933
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049002, 0.0051031
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021505, 0.0020649
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083430, 0.0080112
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016230, 0.0015585
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020282, 0.0021121
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002587, 0.0002694
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014593, 0.0014013
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070152, 0.0073057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041710, upper bound: 0.0041276
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041710, upper bound: 0.0041276
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004196, 0.0004108
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022747, 0.0023236
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051911, 0.0050820
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021416, 0.0021875
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083085, 0.0084868
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016163, 0.0016510
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021486, 0.0021034
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002741, 0.0002683
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014533, 0.0014845
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074317, 0.0072755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052613, upper bound: 0.0051588
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052996, upper bound: 0.0051061
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004198, 0.0004107
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022741, 0.0023242
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051926, 0.0050805
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021409, 0.0021882
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083060, 0.0084893
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016158, 0.0016515
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021492, 0.0021028
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002741, 0.0002682
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014529, 0.0014849
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074339, 0.0072734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049446, upper bound: 0.0047001
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045844, upper bound: 0.0051470
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004016, 0.0003868
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021419, 0.0022238
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049683, 0.0047853
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020165, 0.0020937
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0078234, 0.0081226
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015220, 0.0015802
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020564, 0.0019806
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002623, 0.0002526
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013684, 0.0014208
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0071128, 0.0068508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041517, upper bound: 0.0040843
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041517, upper bound: 0.0040843
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004018, 0.0003867
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021412, 0.0022247
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049703, 0.0047837
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020159, 0.0020945
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0078208, 0.0081258
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015215, 0.0015808
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020572, 0.0019800
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002624, 0.0002526
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013680, 0.0014213
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0071156, 0.0068485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053134, upper bound: 0.0052141
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051769, upper bound: 0.0053448
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004262, 0.0004318
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023908, 0.0023600
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052726, 0.0053413
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022508, 0.0022219
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0087324, 0.0086200
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016988, 0.0016769
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021823, 0.0022107
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002784, 0.0002820
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015274, 0.0015078
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075483, 0.0076467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057947, upper bound: 0.0055195
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058584, upper bound: 0.0054934
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004264, 0.0004317
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023901, 0.0023607
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052741, 0.0053397
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022501, 0.0022225
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0087297, 0.0086225
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016983, 0.0016774
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021829, 0.0022101
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002785, 0.0002819
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015270, 0.0015082
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075505, 0.0076444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053754, upper bound: 0.0051957
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051912, upper bound: 0.0053361
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004194, 0.0004373
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024214, 0.0023220
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051875, 0.0054097
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022796, 0.0021860
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088442, 0.0084810
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017205, 0.0016499
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021471, 0.0022390
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002739, 0.0002856
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015470, 0.0014835
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074266, 0.0077446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051666, upper bound: 0.0050632
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050728, upper bound: 0.0052189
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004298, 0.0004175
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023120, 0.0023798
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053168, 0.0051652
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021766, 0.0022405
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084444, 0.0086923
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016428, 0.0016910
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022006, 0.0021378
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002807, 0.0002727
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014771, 0.0015204
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076116, 0.0073946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0050305, upper bound: 0.0052065
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050813, upper bound: 0.0051803
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004089, 0.0004093
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022666, 0.0022640
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050580, 0.0050637
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021339, 0.0021315
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0082786, 0.0082692
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016105, 0.0016087
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020935, 0.0020958
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002670, 0.0002673
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014481, 0.0014464
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072412, 0.0072494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053649, upper bound: 0.0051370
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051600, upper bound: 0.0052970
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004093, 0.0004094
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022669, 0.0022662
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050629, 0.0050644
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021342, 0.0021335
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0082798, 0.0082772
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016107, 0.0016102
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020955, 0.0020961
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002673, 0.0002674
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014483, 0.0014478
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072481, 0.0072504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052042, upper bound: 0.0049863
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051647, upper bound: 0.0050173
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003846, 0.0003946
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021849, 0.0021294
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0047574, 0.0048813
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020570, 0.0020048
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0079803, 0.0077778
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015525, 0.0015131
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019691, 0.0020203
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002512, 0.0002577
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013959, 0.0013605
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0068108, 0.0069882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054022, upper bound: 0.0051658
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054715, upper bound: 0.0051585
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004014, 0.0003775
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0020904, 0.0022226
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049655, 0.0046702
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019680, 0.0020925
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0076351, 0.0081180
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014853, 0.0015793
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020552, 0.0019330
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002622, 0.0002466
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013355, 0.0014200
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0071087, 0.0066859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044997, upper bound: 0.0050313
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044790, upper bound: 0.0050873
time: 0.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0053015, upper bound: 0.0050747
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052320, upper bound: 0.0050852
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0042029, upper bound: 0.0041238
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0041959, upper bound: 0.0041241
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052368, upper bound: 0.0049434
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051837, upper bound: 0.0049563
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0054106, upper bound: 0.0050198
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052613, upper bound: 0.0051873
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0048905, upper bound: 0.0044826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0048704, upper bound: 0.0045055
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0048374, upper bound: 0.0045026
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0048063, upper bound: 0.0045209
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0045127, upper bound: 0.0051678
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0045420, upper bound: 0.0051308
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051250, upper bound: 0.0050895
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0050935, upper bound: 0.0051043
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051168, upper bound: 0.0051632
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0050851, upper bound: 0.0051858
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0056006, upper bound: 0.0056490
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0056464, upper bound: 0.0056191
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052990, upper bound: 0.0052128
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051565, upper bound: 0.0054030
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0046029, upper bound: 0.0051799
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0046088, upper bound: 0.0051404
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0055249, upper bound: 0.0050961
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0055428, upper bound: 0.0050191
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0055442, upper bound: 0.0050750
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0053340, upper bound: 0.0051590
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0053162, upper bound: 0.0051947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052949, upper bound: 0.0052196
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0041710, upper bound: 0.0041276
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0041710, upper bound: 0.0041276
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052613, upper bound: 0.0051588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052996, upper bound: 0.0051061
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0049446, upper bound: 0.0047001
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0045844, upper bound: 0.0051470
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0041517, upper bound: 0.0040843
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0041517, upper bound: 0.0040843
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0053134, upper bound: 0.0052141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051769, upper bound: 0.0053448
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0057947, upper bound: 0.0055195
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0058584, upper bound: 0.0054934
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0053754, upper bound: 0.0051957
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051912, upper bound: 0.0053361
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051666, upper bound: 0.0050632
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0050728, upper bound: 0.0052189
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0050305, upper bound: 0.0052065
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0050813, upper bound: 0.0051803
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0053649, upper bound: 0.0051370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051600, upper bound: 0.0052970
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0052042, upper bound: 0.0049863
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0051647, upper bound: 0.0050173
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0054022, upper bound: 0.0051658
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0054715, upper bound: 0.0051585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0044997, upper bound: 0.0050313
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.93
Output dim: 4, lower bound: -0.0044790, upper bound: 0.0050873

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003975, 0.0004225
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023392, 0.0022010
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049173, 0.0052259
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022022, 0.0020722
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085438, 0.0080393
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016621, 0.0015640
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020353, 0.0021630
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002596, 0.0002759
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014944, 0.0014062
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070398, 0.0074816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041732, upper bound: 0.0040863
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041664, upper bound: 0.0040865
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003989, 0.0004207
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023292, 0.0022087
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049345, 0.0052037
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021929, 0.0020794
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0085074, 0.0080674
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016550, 0.0015694
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020424, 0.0021538
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002605, 0.0002747
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014881, 0.0014111
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070644, 0.0074498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041028, upper bound: 0.0040961
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040960, upper bound: 0.0040965
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003950, 0.0004193
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023217, 0.0021870
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048861, 0.0051870
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021858, 0.0020590
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084801, 0.0079881
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016497, 0.0015540
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020223, 0.0021469
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002580, 0.0002739
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014833, 0.0013973
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069950, 0.0074258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049743, upper bound: 0.0043036
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045501, upper bound: 0.0046533
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003940, 0.0004123
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022827, 0.0021815
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048736, 0.0050998
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021491, 0.0020538
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083376, 0.0079678
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016220, 0.0015501
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020172, 0.0021108
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002573, 0.0002693
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014584, 0.0013937
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069772, 0.0073011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053829, upper bound: 0.0049614
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053074, upper bound: 0.0049930
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004088, 0.0003944
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021836, 0.0022633
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050566, 0.0048785
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020558, 0.0021308
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0079757, 0.0082669
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015516, 0.0016082
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020929, 0.0020192
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002670, 0.0002576
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013951, 0.0014460
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072391, 0.0069841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041012, upper bound: 0.0040092
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041012, upper bound: 0.0040092
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004108, 0.0004078
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022577, 0.0022748
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050823, 0.0050441
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021256, 0.0021417
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0082464, 0.0083089
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016043, 0.0016164
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021035, 0.0020877
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002683, 0.0002663
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014424, 0.0014534
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072759, 0.0072212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051790, upper bound: 0.0050880
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049887, upper bound: 0.0052421
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004110, 0.0004076
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022568, 0.0022754
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050836, 0.0050420
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021247, 0.0021422
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0082431, 0.0083111
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016036, 0.0016168
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021041, 0.0020869
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002684, 0.0002662
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014418, 0.0014537
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072778, 0.0072182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053834, upper bound: 0.0049904
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0049541, upper bound: 0.0053428
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004197, 0.0004297
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023791, 0.0023239
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051918, 0.0053151
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022398, 0.0021879
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0086896, 0.0084880
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016905, 0.0016513
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021489, 0.0021999
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002741, 0.0002806
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015200, 0.0014847
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074327, 0.0076093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050360, upper bound: 0.0045818
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045862, upper bound: 0.0049418
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004333, 0.0004112
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022766, 0.0023992
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0053602, 0.0050861
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021433, 0.0022588
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083152, 0.0087632
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016176, 0.0017048
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0022185, 0.0021051
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002830, 0.0002685
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014545, 0.0015328
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0076737, 0.0072814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048917, upper bound: 0.0048111
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044148, upper bound: 0.0051155
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003906, 0.0004196
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023231, 0.0021630
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048324, 0.0051901
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021871, 0.0020364
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084852, 0.0079004
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016507, 0.0015369
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020001, 0.0021482
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002551, 0.0002740
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014842, 0.0013819
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069182, 0.0074303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052626, upper bound: 0.0043522
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048749, upper bound: 0.0048310
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003905, 0.0004180
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023143, 0.0021623
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048308, 0.0051705
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021789, 0.0020357
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084532, 0.0078979
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016445, 0.0015364
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019995, 0.0021400
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002551, 0.0002730
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014786, 0.0013815
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069160, 0.0074022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055054, upper bound: 0.0049051
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053101, upper bound: 0.0049832
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004136, 0.0004365
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024167, 0.0022901
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051163, 0.0053991
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022752, 0.0021560
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088269, 0.0083646
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017172, 0.0016272
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021176, 0.0022347
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002701, 0.0002851
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015440, 0.0014631
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0073246, 0.0077295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054292, upper bound: 0.0049797
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054485, upper bound: 0.0049173
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004137, 0.0004363
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024160, 0.0022908
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051179, 0.0053975
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022745, 0.0021567
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088243, 0.0083671
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017167, 0.0016277
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021183, 0.0022340
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002702, 0.0002850
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015435, 0.0014635
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0073269, 0.0077272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041389, upper bound: 0.0040667
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041389, upper bound: 0.0040667
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003941, 0.0004157
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023017, 0.0021823
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048755, 0.0051422
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021669, 0.0020546
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084069, 0.0079709
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016355, 0.0015507
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020180, 0.0021283
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002574, 0.0002715
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014705, 0.0013942
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069799, 0.0073617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050478, upper bound: 0.0044387
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0046469, upper bound: 0.0049397
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003965, 0.0004119
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022805, 0.0021955
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049050, 0.0050950
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021470, 0.0020670
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083297, 0.0080191
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016205, 0.0015600
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020301, 0.0021088
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002590, 0.0002690
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014570, 0.0014027
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070221, 0.0072941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050299, upper bound: 0.0044809
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0046165, upper bound: 0.0049631
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003993, 0.0003895
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021567, 0.0022109
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049395, 0.0048184
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020305, 0.0020815
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0078775, 0.0080754
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015325, 0.0015710
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020444, 0.0019943
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002608, 0.0002544
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013779, 0.0014125
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070715, 0.0068981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052587, upper bound: 0.0050949
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050948, upper bound: 0.0051560
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003992, 0.0003891
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021542, 0.0022102
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049379, 0.0048126
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020281, 0.0020808
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0078681, 0.0080729
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015307, 0.0015705
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020438, 0.0019919
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002607, 0.0002541
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013763, 0.0014121
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070692, 0.0068899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041650, upper bound: 0.0039144
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041650, upper bound: 0.0039144
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004077, 0.0003963
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021943, 0.0022572
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050428, 0.0049023
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020658, 0.0021250
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0080147, 0.0082444
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015592, 0.0016039
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020872, 0.0020290
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002662, 0.0002588
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014019, 0.0014421
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072194, 0.0070183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052757, upper bound: 0.0050853
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050879, upper bound: 0.0051790
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004113, 0.0003928
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021748, 0.0022772
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050875, 0.0048587
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020475, 0.0021439
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0079434, 0.0083174
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015453, 0.0016181
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021057, 0.0020110
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002686, 0.0002565
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013894, 0.0014548
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072833, 0.0069558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049263, upper bound: 0.0046731
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044583, upper bound: 0.0050794
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004066, 0.0004122
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022824, 0.0022513
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050296, 0.0050991
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021488, 0.0021195
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083365, 0.0082228
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016218, 0.0015997
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020817, 0.0021105
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002655, 0.0002692
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014582, 0.0014383
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072005, 0.0073001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055215, upper bound: 0.0048517
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051844, upper bound: 0.0052435
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004067, 0.0004119
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022808, 0.0022516
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050304, 0.0050956
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021473, 0.0021198
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083306, 0.0082241
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016206, 0.0015999
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020821, 0.0021090
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002656, 0.0002690
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014572, 0.0014385
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072017, 0.0072949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055948, upper bound: 0.0048377
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052133, upper bound: 0.0052061
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004127, 0.0004386
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024286, 0.0022854
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051058, 0.0054258
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022865, 0.0021516
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088706, 0.0083473
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017257, 0.0016239
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021132, 0.0022457
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002696, 0.0002865
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015516, 0.0014601
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0073095, 0.0077677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041424, upper bound: 0.0040745
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041424, upper bound: 0.0040745
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004264, 0.0004180
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023147, 0.0023607
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0052741, 0.0051713
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021792, 0.0022225
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084545, 0.0086225
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016447, 0.0016774
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021829, 0.0021404
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002785, 0.0002730
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014788, 0.0015082
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0075505, 0.0074034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041369, upper bound: 0.0040746
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041369, upper bound: 0.0040746
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004195, 0.0004373
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0024214, 0.0023227
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0051892, 0.0054097
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0022797, 0.0021867
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0088442, 0.0084837
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0017206, 0.0016504
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021478, 0.0022391
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002740, 0.0002856
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0015470, 0.0014839
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0074289, 0.0077447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049277, upper bound: 0.0051158
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049727, upper bound: 0.0050848
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004102, 0.0003975
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022009, 0.0022712
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050741, 0.0049170
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020720, 0.0021382
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0080386, 0.0082955
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015638, 0.0016138
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021001, 0.0020351
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002679, 0.0002596
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014061, 0.0014510
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072642, 0.0070392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049982, upper bound: 0.0049967
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049181, upper bound: 0.0051725
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003943, 0.0004143
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022937, 0.0021833
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048778, 0.0051244
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021594, 0.0020555
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083777, 0.0079745
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016298, 0.0015514
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020189, 0.0021210
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002575, 0.0002705
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014654, 0.0013949
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069831, 0.0073362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053281, upper bound: 0.0049905
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051548, upper bound: 0.0051008
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004089, 0.0003948
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021859, 0.0022640
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050580, 0.0048835
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020579, 0.0021315
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0079839, 0.0082692
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015532, 0.0016087
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020935, 0.0020212
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002670, 0.0002578
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013965, 0.0014464
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072412, 0.0069913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040915, upper bound: 0.0040104
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040915, upper bound: 0.0040104
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003983, 0.0004183
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023159, 0.0022055
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0049273, 0.0051741
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021804, 0.0020764
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084590, 0.0080556
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016456, 0.0015671
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020394, 0.0021415
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002601, 0.0002732
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014796, 0.0014091
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070541, 0.0074073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041257, upper bound: 0.0039797
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041256, upper bound: 0.0039853
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003640, 0.0003744
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0020732, 0.0020153
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0045024, 0.0046318
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019518, 0.0018973
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0075724, 0.0073609
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014731, 0.0014320
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0018635, 0.0019171
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002377, 0.0002445
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013245, 0.0012875
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0064457, 0.0066310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047428, upper bound: 0.0045098
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047230, upper bound: 0.0045615
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003644, 0.0003748
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0020754, 0.0020177
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0045079, 0.0046367
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019539, 0.0018996
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0075804, 0.0073698
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014747, 0.0014337
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0018658, 0.0019191
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002380, 0.0002448
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013259, 0.0012891
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0064536, 0.0066380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050287, upper bound: 0.0044916
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048820, upper bound: 0.0047441
time: 0.85 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041732, upper bound: 0.0040863
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041664, upper bound: 0.0040865
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041028, upper bound: 0.0040961
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0040960, upper bound: 0.0040965
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049743, upper bound: 0.0043036
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0045501, upper bound: 0.0046533
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0053829, upper bound: 0.0049614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0053074, upper bound: 0.0049930
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041012, upper bound: 0.0040092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041012, upper bound: 0.0040092
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0051790, upper bound: 0.0050880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049887, upper bound: 0.0052421
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0053834, upper bound: 0.0049904
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049541, upper bound: 0.0053428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0050360, upper bound: 0.0045818
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0045862, upper bound: 0.0049418
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0048917, upper bound: 0.0048111
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0044148, upper bound: 0.0051155
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0052626, upper bound: 0.0043522
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0048749, upper bound: 0.0048310
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0055054, upper bound: 0.0049051
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0053101, upper bound: 0.0049832
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0054292, upper bound: 0.0049797
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0054485, upper bound: 0.0049173
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041389, upper bound: 0.0040667
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041389, upper bound: 0.0040667
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0050478, upper bound: 0.0044387
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0046469, upper bound: 0.0049397
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0050299, upper bound: 0.0044809
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0046165, upper bound: 0.0049631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0052587, upper bound: 0.0050949
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0050948, upper bound: 0.0051560
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041650, upper bound: 0.0039144
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041650, upper bound: 0.0039144
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0052757, upper bound: 0.0050853
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0050879, upper bound: 0.0051790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049263, upper bound: 0.0046731
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0044583, upper bound: 0.0050794
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0055215, upper bound: 0.0048517
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0051844, upper bound: 0.0052435
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0055948, upper bound: 0.0048377
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0052133, upper bound: 0.0052061
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041424, upper bound: 0.0040745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041424, upper bound: 0.0040745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041369, upper bound: 0.0040746
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041369, upper bound: 0.0040746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049277, upper bound: 0.0051158
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049727, upper bound: 0.0050848
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049982, upper bound: 0.0049967
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0049181, upper bound: 0.0051725
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0053281, upper bound: 0.0049905
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0051548, upper bound: 0.0051008
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0040915, upper bound: 0.0040104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0040915, upper bound: 0.0040104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041257, upper bound: 0.0039797
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0041256, upper bound: 0.0039853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0047428, upper bound: 0.0045098
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0047230, upper bound: 0.0045615
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0050287, upper bound: 0.0044916
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.08
Output dim: 4, lower bound: -0.0048820, upper bound: 0.0047441

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003939, 0.0004136
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022902, 0.0021807
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048720, 0.0051167
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021562, 0.0020531
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083651, 0.0079652
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016273, 0.0015495
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020165, 0.0021178
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002572, 0.0002701
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014632, 0.0013932
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069749, 0.0073251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051369, upper bound: 0.0043595
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0046471, upper bound: 0.0046718
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003955, 0.0004121
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022820, 0.0021897
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048921, 0.0050982
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021484, 0.0020615
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083350, 0.0079980
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016215, 0.0015559
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020248, 0.0021101
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002583, 0.0002692
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014579, 0.0013990
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0070036, 0.0072988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040268, upper bound: 0.0039819
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040268, upper bound: 0.0039819
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004108, 0.0003932
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021771, 0.0022748
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050823, 0.0048639
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020497, 0.0021417
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0079519, 0.0083089
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015470, 0.0016164
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0021035, 0.0020131
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002683, 0.0002568
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013909, 0.0014534
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072759, 0.0069633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047271, upper bound: 0.0046155
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0042719, upper bound: 0.0049699
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003626, 0.0003769
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0020868, 0.0020076
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0044853, 0.0046620
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019646, 0.0018901
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0076219, 0.0073329
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014828, 0.0014265
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0018564, 0.0019296
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002368, 0.0002461
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013332, 0.0012826
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0064212, 0.0066743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047218, upper bound: 0.0043878
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047019, upper bound: 0.0044129
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003800, 0.0003592
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0019890, 0.0021038
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0047002, 0.0044437
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0018726, 0.0019807
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0072649, 0.0076842
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014133, 0.0014949
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019454, 0.0018392
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002482, 0.0002346
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0012707, 0.0013441
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0067289, 0.0063617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045162, upper bound: 0.0047674
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043304, upper bound: 0.0049359
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003414, 0.0003841
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021267, 0.0018905
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0042236, 0.0047512
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020022, 0.0017798
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0077677, 0.0069050
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015111, 0.0013433
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0017481, 0.0019665
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002230, 0.0002508
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013587, 0.0012078
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0060465, 0.0068020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052247, upper bound: 0.0042882
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049890, upper bound: 0.0043182
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003905, 0.0004181
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023150, 0.0021622
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048306, 0.0051721
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021795, 0.0020356
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084557, 0.0078975
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016450, 0.0015364
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019994, 0.0021407
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002550, 0.0002731
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014790, 0.0013814
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069156, 0.0074044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041724, upper bound: 0.0038783
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041724, upper bound: 0.0038783
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003906, 0.0004180
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023143, 0.0021629
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048321, 0.0051703
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021788, 0.0020363
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084528, 0.0079000
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016444, 0.0015369
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020000, 0.0021400
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002551, 0.0002730
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014785, 0.0013818
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069178, 0.0074019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040721, upper bound: 0.0039131
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0040721, upper bound: 0.0039131
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003929, 0.0004159
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0023030, 0.0021755
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048604, 0.0051452
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021682, 0.0020482
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0084117, 0.0079461
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016364, 0.0015458
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020117, 0.0021296
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002566, 0.0002716
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014713, 0.0013899
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069582, 0.0073659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051693, upper bound: 0.0043090
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047328, upper bound: 0.0047092
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003931, 0.0004146
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022958, 0.0021764
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048623, 0.0051290
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021614, 0.0020490
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083853, 0.0079493
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016313, 0.0015465
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020125, 0.0021229
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002567, 0.0002708
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014667, 0.0013905
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069610, 0.0073428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0051976, upper bound: 0.0042622
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047410, upper bound: 0.0046268
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004052, 0.0003993
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022109, 0.0022436
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050124, 0.0049394
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020815, 0.0021122
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0080754, 0.0081946
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015710, 0.0015942
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020746, 0.0020444
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002646, 0.0002608
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014125, 0.0014334
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0071758, 0.0070714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050138, upper bound: 0.0044447
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045706, upper bound: 0.0048181
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0004076, 0.0003964
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021949, 0.0022571
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0050425, 0.0049037
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020664, 0.0021249
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0080169, 0.0082439
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015596, 0.0016038
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020871, 0.0020296
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002662, 0.0002589
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014023, 0.0014420
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0072190, 0.0070202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041317, upper bound: 0.0038957
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0041317, upper bound: 0.0038957
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003582, 0.0003803
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021057, 0.0019835
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0044313, 0.0047044
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019825, 0.0018673
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0076912, 0.0072446
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014962, 0.0014094
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0018341, 0.0019471
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002340, 0.0002484
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013453, 0.0012672
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0063439, 0.0067350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051250, upper bound: 0.0042978
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048844, upper bound: 0.0044624
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003761, 0.0003638
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0020146, 0.0020822
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0046518, 0.0045008
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0018967, 0.0019603
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0073583, 0.0076052
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014315, 0.0014795
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019254, 0.0018629
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002456, 0.0002376
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0012871, 0.0013303
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0066597, 0.0064435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047853, upper bound: 0.0047204
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045225, upper bound: 0.0048468
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003583, 0.0003808
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021086, 0.0019838
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0044321, 0.0047109
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019852, 0.0018677
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0077018, 0.0072459
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014983, 0.0014096
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0018344, 0.0019498
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002340, 0.0002487
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013472, 0.0012674
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0063451, 0.0067442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049506, upper bound: 0.0042192
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049171, upper bound: 0.0042388
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003759, 0.0003636
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0020130, 0.0020814
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0046501, 0.0044972
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0018951, 0.0019596
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0073524, 0.0076024
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014303, 0.0014790
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0019247, 0.0018614
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002455, 0.0002374
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0012861, 0.0013298
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0066573, 0.0064384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045926, upper bound: 0.0045595
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045643, upper bound: 0.0045830
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003943, 0.0004144
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0022943, 0.0021832
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0048775, 0.0051258
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0021600, 0.0020554
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0083800, 0.0079742
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0016302, 0.0015513
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0020188, 0.0021215
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002575, 0.0002706
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0014658, 0.0013948
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0069828, 0.0073382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050446, upper bound: 0.0043158
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0046737, upper bound: 0.0047244
time: 0.82 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0051369, upper bound: 0.0043595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0046471, upper bound: 0.0046718
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0040268, upper bound: 0.0039819
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0040268, upper bound: 0.0039819
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0047271, upper bound: 0.0046155
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0042719, upper bound: 0.0049699
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0047218, upper bound: 0.0043878
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0047019, upper bound: 0.0044129
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0045162, upper bound: 0.0047674
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0043304, upper bound: 0.0049359
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0052247, upper bound: 0.0042882
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0049890, upper bound: 0.0043182
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0041724, upper bound: 0.0038783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0041724, upper bound: 0.0038783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0040721, upper bound: 0.0039131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0040721, upper bound: 0.0039131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0051693, upper bound: 0.0043090
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0047328, upper bound: 0.0047092
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0051976, upper bound: 0.0042622
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0047410, upper bound: 0.0046268
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0050138, upper bound: 0.0044447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0045706, upper bound: 0.0048181
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0041317, upper bound: 0.0038957
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0041317, upper bound: 0.0038957
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0051250, upper bound: 0.0042978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0048844, upper bound: 0.0044624
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0047853, upper bound: 0.0047204
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0045225, upper bound: 0.0048468
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0049506, upper bound: 0.0042192
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0049171, upper bound: 0.0042388
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0045926, upper bound: 0.0045595
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0045643, upper bound: 0.0045830
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0050446, upper bound: 0.0043158
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 4, lower bound: -0.0046737, upper bound: 0.0047244

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003416, 0.0003844
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021283, 0.0018914
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0042255, 0.0047549
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0020037, 0.0017806
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0077737, 0.0069082
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0015123, 0.0013439
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0017489, 0.0019680
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002231, 0.0002510
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013597, 0.0012084
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0060493, 0.0068072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038558, upper bound: 0.0032788
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038558, upper bound: 0.0032788
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042388, -0.0037099, -0.0042388, -0.0037099, -0.0003440, 0.0003810
1: 0.0002894, 0.0032183, 0.0002894, 0.0032183, -0.0021096, 0.0019049
2: 0.0077762, 0.0143195, 0.0077762, 0.0143195, -0.0042556, 0.0047131
3: 0.0013001, 0.0040575, 0.0013001, 0.0040575, -0.0019861, 0.0017933
4: 1.0017941, 1.0124917, 1.0017941, 1.0124917, -0.0077054, 0.0069575
5: 0.0025904, 0.0046715, 0.0025904, 0.0046715, -0.0014990, 0.0013535
6: -0.0118223, -0.0091140, -0.0118223, -0.0091140, -0.0017614, 0.0019507
7: -0.0103114, -0.0099659, -0.0103114, -0.0099659, -0.0002247, 0.0002488
8: -0.0045570, -0.0026859, -0.0045570, -0.0026859, -0.0013478, 0.0012170
9: -0.0047248, 0.0046428, -0.0047248, 0.0046428, -0.0060925, 0.0067474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038344, upper bound: 0.0032625
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038344, upper bound: 0.0032625
time: 0.76 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.97 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 4, lower bound: -0.0038558, upper bound: 0.0032788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 4, lower bound: -0.0038558, upper bound: 0.0032788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 4, lower bound: -0.0038344, upper bound: 0.0032625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 4, lower bound: -0.0038344, upper bound: 0.0032625

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.30 + 320.48 = 323.78 seconds
