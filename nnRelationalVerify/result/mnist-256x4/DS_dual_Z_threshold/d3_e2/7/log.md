## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0036601199999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243)
1: (-0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566)
2: (0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881)
3: (-0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0382411, 0.0382411)
4: (-0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348)
5: (0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758)
6: (0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265)
7: (-0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264)
8: (0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744)
9: (0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148966, 0.0148966)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.05 + 3.32 = 5.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0040668, upper bound: 0.0040668

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040422, upper bound: 0.0040583
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040583, upper bound: 0.0040422
time: 3.02 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.57
Output dim: 2, lower bound: -0.0040422, upper bound: 0.0040583
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.57
Output dim: 2, lower bound: -0.0040583, upper bound: 0.0040422

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0382317, 0.0382444
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148971, 0.0148957

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039366, upper bound: 0.0040039
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039846, upper bound: 0.0039568
time: 2.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0382444, 0.0382317
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148957, 0.0148971

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039568, upper bound: 0.0039846
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0040041, upper bound: 0.0039367
time: 3.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 7.49 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.49
Output dim: 2, lower bound: -0.0039366, upper bound: 0.0040039
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.49
Output dim: 2, lower bound: -0.0039846, upper bound: 0.0039568
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.49
Output dim: 2, lower bound: -0.0039568, upper bound: 0.0039846
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.49
Output dim: 2, lower bound: -0.0040041, upper bound: 0.0039367

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380534, 0.0381771
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148895, 0.0148759

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036663, upper bound: 0.0039079
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038584, upper bound: 0.0037339
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0382317, 0.0380662
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148773, 0.0148957

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037219, upper bound: 0.0038744
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038920, upper bound: 0.0036788
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380662, 0.0381615
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148878, 0.0148773

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036788, upper bound: 0.0038916
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038741, upper bound: 0.0037219
time: 2.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0382444, 0.0380534
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148759, 0.0148971

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037339, upper bound: 0.0038584
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039083, upper bound: 0.0036663
time: 2.62 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 7.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0036663, upper bound: 0.0039079
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0038584, upper bound: 0.0037339
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0037219, upper bound: 0.0038744
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0038920, upper bound: 0.0036788
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0036788, upper bound: 0.0038916
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0038741, upper bound: 0.0037219
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0037339, upper bound: 0.0038584
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.30
Output dim: 2, lower bound: -0.0039083, upper bound: 0.0036663

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0365752, 0.0371645
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147685, 0.0147038

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0370794, 0.0366988
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147174, 0.0147592

Time for backsubstitution: 2.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0367622, 0.0370883
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147602, 0.0147237

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0372665, 0.0365879
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147052, 0.0147790

Time for backsubstitution: 2.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0365879, 0.0371537
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147674, 0.0147052

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030818
time: 29.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030822
time: 18.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0370883, 0.0366833
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147157, 0.0147602

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0367750, 0.0370794
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147592, 0.0147251

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030822
time: 14.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030822
time: 21.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0372753, 0.0365752
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147038, 0.0147800

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744
time: 1.83 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029744, upper bound: 0.0030862
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030818, upper bound: 0.0029774
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030818
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030822
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030822
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0029774, upper bound: 0.0030822
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.59
Output dim: 2, lower bound: -0.0030858, upper bound: 0.0029744

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.37 + 177.71 = 183.07 seconds
