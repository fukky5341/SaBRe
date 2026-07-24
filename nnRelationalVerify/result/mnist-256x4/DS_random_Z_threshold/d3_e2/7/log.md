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
execution time: IAR + RelationalAnalysis = 0.86 + 3.19 = 4.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0040668, upper bound: 0.0040668

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039497
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039492
time: 2.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.75
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039497
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.75
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039492

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381176, 0.0381467
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148864, 0.0148832

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039492
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039492
time: 2.75 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381467, 0.0382411
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148966, 0.0148864

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039414, upper bound: 0.0039442
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039442, upper bound: 0.0039414
time: 2.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039492
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 2, lower bound: -0.0039497, upper bound: 0.0039492
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 2, lower bound: -0.0039414, upper bound: 0.0039442
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 2, lower bound: -0.0039442, upper bound: 0.0039414

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380434, 0.0380708
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148781, 0.0148751

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039049, upper bound: 0.0039497
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039496, upper bound: 0.0039048
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380417, 0.0380700
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148780, 0.0148749

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038663, upper bound: 0.0038808
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0038658
time: 2.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381628, 0.0382641
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148987, 0.0148877

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036618, upper bound: 0.0036654
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036618, upper bound: 0.0036654
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381700, 0.0382568
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148979, 0.0148885

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038996, upper bound: 0.0039414
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039442, upper bound: 0.0038992
time: 2.61 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.09 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0039049, upper bound: 0.0039497
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0039496, upper bound: 0.0039048
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0038663, upper bound: 0.0038808
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0038658
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0036618, upper bound: 0.0036654
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0036618, upper bound: 0.0036654
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0038996, upper bound: 0.0039414
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 2, lower bound: -0.0039442, upper bound: 0.0038992

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380156, 0.0380466
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148756, 0.0148722

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031898, upper bound: 0.0032025
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031898, upper bound: 0.0032025
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380204, 0.0380430
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148752, 0.0148727

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036729, upper bound: 0.0036346
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036346, upper bound: 0.0036346
time: 2.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380237, 0.0380609
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148770, 0.0148730

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037724, upper bound: 0.0038708
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038560, upper bound: 0.0037776
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380417, 0.0380521
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148761, 0.0148749

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035409, upper bound: 0.0035409
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035409, upper bound: 0.0035409
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380900, 0.0382713
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148993, 0.0148795

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035627, upper bound: 0.0035926
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035892, upper bound: 0.0035672
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381628, 0.0381913
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148905, 0.0148877

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035439, upper bound: 0.0036516
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036479, upper bound: 0.0035452
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381424, 0.0382323
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148954, 0.0148856

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037528, upper bound: 0.0038038
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037650, upper bound: 0.0037871
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381460, 0.0382290
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148950, 0.0148860

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036668, upper bound: 0.0036288
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036668, upper bound: 0.0036288
time: 2.04 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0031898, upper bound: 0.0032025
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0031898, upper bound: 0.0032025
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0036729, upper bound: 0.0036346
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0036346, upper bound: 0.0036346
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0037724, upper bound: 0.0038708
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0038560, upper bound: 0.0037776
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0035409, upper bound: 0.0035409
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0035409, upper bound: 0.0035409
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0035627, upper bound: 0.0035926
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0035892, upper bound: 0.0035672
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0035439, upper bound: 0.0036516
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0036479, upper bound: 0.0035452
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0037528, upper bound: 0.0038038
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0037650, upper bound: 0.0037871
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0036668, upper bound: 0.0036288
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 2, lower bound: -0.0036668, upper bound: 0.0036288

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0379639, 0.0380100
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148716, 0.0148666

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036556, upper bound: 0.0036305
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036683, upper bound: 0.0036222
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0370278, 0.0374585
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148057, 0.0147584

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037526, upper bound: 0.0038708
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037724, upper bound: 0.0038310
time: 2.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0374437, 0.0370650
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147625, 0.0148041

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037366, upper bound: 0.0037438
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038211, upper bound: 0.0037414
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0379660, 0.0381699
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148881, 0.0148658

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037528, upper bound: 0.0038013
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037501, upper bound: 0.0038038
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381424, 0.0380559
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148756, 0.0148856

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036756, upper bound: 0.0037759
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037539, upper bound: 0.0036906
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0380900, 0.0382013
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148920, 0.0148798

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035704, upper bound: 0.0035568
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035946, upper bound: 0.0035329
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381460, 0.0381730
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148889, 0.0148860

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036503, upper bound: 0.0036244
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036624, upper bound: 0.0036152
time: 2.43 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.63 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0036556, upper bound: 0.0036305
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0036683, upper bound: 0.0036222
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0037526, upper bound: 0.0038708
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0037724, upper bound: 0.0038310
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0037366, upper bound: 0.0037438
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0038211, upper bound: 0.0037414
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0037528, upper bound: 0.0038013
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0037501, upper bound: 0.0038038
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0036756, upper bound: 0.0037759
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0037539, upper bound: 0.0036906
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0035704, upper bound: 0.0035568
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0035946, upper bound: 0.0035329
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0036503, upper bound: 0.0036244
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.63
Output dim: 2, lower bound: -0.0036624, upper bound: 0.0036152

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0379675, 0.0380003
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148705, 0.0148669

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036342, upper bound: 0.0035914
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036371, upper bound: 0.0035913
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0370007, 0.0374367
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148034, 0.0147555

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033792, upper bound: 0.0034503
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033792, upper bound: 0.0034503
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0370050, 0.0374314
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148028, 0.0147560

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037724, upper bound: 0.0038294
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0038310
time: 2.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0373658, 0.0370217
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147553, 0.0147931

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036113, upper bound: 0.0036771
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037482, upper bound: 0.0036180
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0374003, 0.0369750
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147502, 0.0147969

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037986, upper bound: 0.0037313
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038104, upper bound: 0.0037279
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0378000, 0.0379971
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148689, 0.0148474

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036598, upper bound: 0.0036918
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036611, upper bound: 0.0036891
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0377934, 0.0380152
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148709, 0.0148466

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037179, upper bound: 0.0037727
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037195, upper bound: 0.0037664
time: 2.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0371369, 0.0374398
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148038, 0.0147709

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035587, upper bound: 0.0036984
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036016, upper bound: 0.0036686
time: 2.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0375254, 0.0370477
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147608, 0.0148135

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033881, upper bound: 0.0033562
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033881, upper bound: 0.0033562
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0381468, 0.0381596
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148876, 0.0148862

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034482, upper bound: 0.0034918
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035444, upper bound: 0.0034130
time: 1.71 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.59 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0036342, upper bound: 0.0035914
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0036371, upper bound: 0.0035913
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0033792, upper bound: 0.0034503
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0033792, upper bound: 0.0034503
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0037724, upper bound: 0.0038294
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0037718, upper bound: 0.0038310
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0036113, upper bound: 0.0036771
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0037482, upper bound: 0.0036180
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0037986, upper bound: 0.0037313
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0038104, upper bound: 0.0037279
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0036598, upper bound: 0.0036918
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0036611, upper bound: 0.0036891
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0037179, upper bound: 0.0037727
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0037195, upper bound: 0.0037664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0035587, upper bound: 0.0036984
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0036016, upper bound: 0.0036686
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0033881, upper bound: 0.0033562
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0033881, upper bound: 0.0033562
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0034482, upper bound: 0.0034918
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 2, lower bound: -0.0035444, upper bound: 0.0034130

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0368412, 0.0372553
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147835, 0.0147380

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035859, upper bound: 0.0037164
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036527, upper bound: 0.0036190
time: 2.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0368289, 0.0372750
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147857, 0.0147367

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037668, upper bound: 0.0038261
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037667, upper bound: 0.0038262
time: 2.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0368631, 0.0366256
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147136, 0.0147397

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036553, upper bound: 0.0036442
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036714, upper bound: 0.0036200
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0369663, 0.0365190
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147019, 0.0147510

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035679, upper bound: 0.0034788
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035773, upper bound: 0.0034716
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0373861, 0.0369797
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147508, 0.0147955

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034030, upper bound: 0.0033581
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034030, upper bound: 0.0033581
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0374050, 0.0369698
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147498, 0.0147975

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035541, upper bound: 0.0034797
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035541, upper bound: 0.0034797
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0377818, 0.0379863
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148677, 0.0148454

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0378000, 0.0379789
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148669, 0.0148474

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0376839, 0.0379553
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148634, 0.0148336

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036120, upper bound: 0.0036968
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036411, upper bound: 0.0036703
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0377339, 0.0378904
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148562, 0.0148391

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037048, upper bound: 0.0037603
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037048, upper bound: 0.0037483
time: 2.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0366404, 0.0370319
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147604, 0.0147176

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035278, upper bound: 0.0036673
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035290, upper bound: 0.0036616
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0367535, 0.0369421
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147505, 0.0147300

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034777, upper bound: 0.0035604
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034777, upper bound: 0.0035604
time: 2.97 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 6.84 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0035859, upper bound: 0.0037164
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0036527, upper bound: 0.0036190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0037668, upper bound: 0.0038261
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0037667, upper bound: 0.0038262
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0036553, upper bound: 0.0036442
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0036714, upper bound: 0.0036200
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0035679, upper bound: 0.0034788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0035773, upper bound: 0.0034716
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0034030, upper bound: 0.0033581
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0034030, upper bound: 0.0033581
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0035541, upper bound: 0.0034797
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0035541, upper bound: 0.0034797
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0031782, upper bound: 0.0031919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0036120, upper bound: 0.0036968
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0036411, upper bound: 0.0036703
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0037048, upper bound: 0.0037603
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0037048, upper bound: 0.0037483
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0035278, upper bound: 0.0036673
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0035290, upper bound: 0.0036616
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0034777, upper bound: 0.0035604
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.84
Output dim: 2, lower bound: -0.0034777, upper bound: 0.0035604

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0356830, 0.0363613
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0146892, 0.0146147

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033376, upper bound: 0.0034252
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033376, upper bound: 0.0034252
time: 2.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0368476, 0.0372933
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147878, 0.0147389

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034443, upper bound: 0.0034879
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034443, upper bound: 0.0034879
time: 2.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0368472, 0.0372828
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147867, 0.0147388

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036286, upper bound: 0.0037930
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037341, upper bound: 0.0036801
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0365452, 0.0362878
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0146781, 0.0147063

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036495, upper bound: 0.0036062
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036584, upper bound: 0.0036030
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0372010, 0.0375760
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148214, 0.0147801

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035639, upper bound: 0.0036657
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035784, upper bound: 0.0036405
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0373264, 0.0374707
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148098, 0.0147939

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035224, upper bound: 0.0035517
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035224, upper bound: 0.0035517
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0377208, 0.0378896
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148565, 0.0148380

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035726, upper bound: 0.0037296
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036719, upper bound: 0.0036034
time: 2.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0377336, 0.0378771
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148551, 0.0148394

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036309, upper bound: 0.0037374
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037047, upper bound: 0.0036595
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0365434, 0.0369840
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147532, 0.0147050

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031475, upper bound: 0.0032350
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031475, upper bound: 0.0032350
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0365935, 0.0369490
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147493, 0.0147105

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033457, upper bound: 0.0035412
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033971, upper bound: 0.0034514
time: 2.46 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 6.07 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0033376, upper bound: 0.0034252
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0033376, upper bound: 0.0034252
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0034443, upper bound: 0.0034879
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0034443, upper bound: 0.0034879
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0036286, upper bound: 0.0037930
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0037341, upper bound: 0.0036801
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0036495, upper bound: 0.0036062
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0036584, upper bound: 0.0036030
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0035639, upper bound: 0.0036657
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0035784, upper bound: 0.0036405
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0035224, upper bound: 0.0035517
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0035224, upper bound: 0.0035517
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0035726, upper bound: 0.0037296
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0036719, upper bound: 0.0036034
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0036309, upper bound: 0.0037374
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0037047, upper bound: 0.0036595
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0031475, upper bound: 0.0032350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0031475, upper bound: 0.0032350
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0033457, upper bound: 0.0035412
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 6.07
Output dim: 2, lower bound: -0.0033971, upper bound: 0.0034514

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0360088, 0.0367296
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147232, 0.0146440

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035880, upper bound: 0.0037580
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035969, upper bound: 0.0037457
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0362845, 0.0364444
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0146919, 0.0146743

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035075, upper bound: 0.0035836
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036244, upper bound: 0.0034134
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0369054, 0.0373412
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147945, 0.0147465

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035383, upper bound: 0.0036553
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035531, upper bound: 0.0036383
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0369024, 0.0373592
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147959, 0.0147460

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032017, upper bound: 0.0032803
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032017, upper bound: 0.0032803
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0371805, 0.0370740
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147645, 0.0147765

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032880, upper bound: 0.0032429
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032880, upper bound: 0.0032429
time: 2.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0367423, 0.0373133
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147875, 0.0147247

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032806, upper bound: 0.0033274
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032806, upper bound: 0.0033274
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0371307, 0.0368840
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0147404, 0.0147674

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035088, upper bound: 0.0035315
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035817, upper bound: 0.0035315
time: 2.34 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 6.38 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0035880, upper bound: 0.0037580
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0035969, upper bound: 0.0037457
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0035075, upper bound: 0.0035836
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0036244, upper bound: 0.0034134
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0035383, upper bound: 0.0036553
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0035531, upper bound: 0.0036383
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0032017, upper bound: 0.0032803
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0032017, upper bound: 0.0032803
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0032880, upper bound: 0.0032429
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0032880, upper bound: 0.0032429
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0032806, upper bound: 0.0033274
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0032806, upper bound: 0.0033274
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0035088, upper bound: 0.0035315
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.38
Output dim: 2, lower bound: -0.0035817, upper bound: 0.0035315

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0356874, 0.0364717
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0146966, 0.0146105

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033326, upper bound: 0.0034696
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033326, upper bound: 0.0034696
time: 2.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243
1: -0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566
2: 0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881
3: -0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0357196, 0.0364082
4: -0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348
5: 0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758
6: 0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265
7: -0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264
8: 0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744
9: 0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0146896, 0.0146140

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032964, upper bound: 0.0034063
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032964, upper bound: 0.0034063
time: 2.41 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 5.32 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.32
Output dim: 2, lower bound: -0.0033326, upper bound: 0.0034696
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.32
Output dim: 2, lower bound: -0.0033326, upper bound: 0.0034696
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.32
Output dim: 2, lower bound: -0.0032964, upper bound: 0.0034063
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.32
Output dim: 2, lower bound: -0.0032964, upper bound: 0.0034063

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.04 + 360.08 = 364.12 seconds
