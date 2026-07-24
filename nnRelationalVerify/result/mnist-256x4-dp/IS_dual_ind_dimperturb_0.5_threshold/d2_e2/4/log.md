## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0061821


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9810681, 0.9898579, 0.9810681, 0.9898579, -0.0087897, 0.0087897)
1: (-0.0045276, -0.0037911, -0.0045276, -0.0037911, -0.0007365, 0.0007365)
2: (0.0100368, 0.0139400, 0.0100368, 0.0139400, -0.0039032, 0.0039032)
3: (-0.0077147, -0.0058414, -0.0077147, -0.0058414, -0.0018733, 0.0018733)
4: (0.0024705, 0.0036442, 0.0024705, 0.0036442, -0.0011737, 0.0011737)
5: (0.0115831, 0.0208257, 0.0115831, 0.0208257, -0.0092426, 0.0092426)
6: (-0.0026451, -0.0013991, -0.0026451, -0.0013991, -0.0012460, 0.0012460)
7: (-0.0099813, -0.0067575, -0.0099813, -0.0067575, -0.0032238, 0.0032238)
8: (-0.0048132, -0.0026106, -0.0048132, -0.0026106, -0.0022026, 0.0022026)
9: (0.0017514, 0.0037173, 0.0017514, 0.0037173, -0.0019658, 0.0019658)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.81 = 3.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0071007, upper bound: 0.0071007

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068709, upper bound: 0.0068574
time: 0.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068567, upper bound: 0.0068567
time: 1.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.19 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -0.0068709, upper bound: 0.0068574
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -0.0068567, upper bound: 0.0068567

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9811447, 0.9897113, 0.9810681, 0.9898579, -0.0087132, 0.0086432
1: -0.0045265, -0.0038276, -0.0045276, -0.0037911, -0.0007354, 0.0007000
2: 0.0102304, 0.0139341, 0.0100368, 0.0139400, -0.0037096, 0.0038973
3: -0.0077108, -0.0059296, -0.0077147, -0.0058414, -0.0018694, 0.0017851
4: 0.0025080, 0.0036379, 0.0024705, 0.0036442, -0.0011362, 0.0011674
5: 0.0118266, 0.0207648, 0.0115831, 0.0208257, -0.0089991, 0.0091817
6: -0.0026432, -0.0014609, -0.0026451, -0.0013991, -0.0012441, 0.0011842
7: -0.0099764, -0.0069174, -0.0099813, -0.0067575, -0.0032189, 0.0030639
8: -0.0048106, -0.0026352, -0.0048132, -0.0026106, -0.0022000, 0.0021780
9: 0.0018489, 0.0037143, 0.0017514, 0.0037173, -0.0018683, 0.0019629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068422
time: 0.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068422
time: 1.06 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9768384, 0.9896186, 0.9810858, 0.9898040, -0.0129656, 0.0085328
1: -0.0045891, -0.0038401, -0.0045274, -0.0038045, -0.0007846, 0.0006872
2: 0.0103528, 0.0142659, 0.0101080, 0.0139386, -0.0035858, 0.0041579
3: -0.0079290, -0.0059853, -0.0077138, -0.0058738, -0.0020552, 0.0017285
4: 0.0025316, 0.0039928, 0.0024843, 0.0036427, -0.0011111, 0.0015086
5: 0.0119805, 0.0241942, 0.0116726, 0.0208117, -0.0088312, 0.0125216
6: -0.0027491, -0.0015000, -0.0026447, -0.0014218, -0.0013273, 0.0011447
7: -0.0102504, -0.0070185, -0.0099801, -0.0068163, -0.0034342, 0.0029617
8: -0.0049547, -0.0012483, -0.0048126, -0.0026163, -0.0023385, 0.0035643
9: 0.0019106, 0.0038814, 0.0017873, 0.0037166, -0.0018060, 0.0020941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068567
time: 0.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068567
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068422
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068422
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068567
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 0, lower bound: -0.0068422, upper bound: 0.0068567

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9811447, 0.9897113, 0.9811447, 0.9897113, -0.0085666, 0.0085666
1: -0.0045265, -0.0038276, -0.0045265, -0.0038276, -0.0006989, 0.0006989
2: 0.0102304, 0.0139341, 0.0102304, 0.0139341, -0.0037037, 0.0037037
3: -0.0077108, -0.0059296, -0.0077108, -0.0059296, -0.0017813, 0.0017813
4: 0.0025080, 0.0036379, 0.0025080, 0.0036379, -0.0011299, 0.0011299
5: 0.0118266, 0.0207648, 0.0118266, 0.0207648, -0.0089382, 0.0089382
6: -0.0026432, -0.0014609, -0.0026432, -0.0014609, -0.0011823, 0.0011823
7: -0.0099764, -0.0069174, -0.0099764, -0.0069174, -0.0030590, 0.0030590
8: -0.0048106, -0.0026352, -0.0048106, -0.0026352, -0.0021754, 0.0021754
9: 0.0018489, 0.0037143, 0.0018489, 0.0037143, -0.0018654, 0.0018654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067246, upper bound: 0.0066588
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067165, upper bound: 0.0067061
time: 0.91 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9811447, 0.9897113, 0.9768384, 0.9896186, -0.0084739, 0.0128729
1: -0.0045265, -0.0038276, -0.0045891, -0.0038401, -0.0006864, 0.0007615
2: 0.0102304, 0.0139341, 0.0103528, 0.0142659, -0.0040355, 0.0035813
3: -0.0077108, -0.0059296, -0.0079290, -0.0059853, -0.0017256, 0.0019995
4: 0.0025080, 0.0036379, 0.0025316, 0.0039928, -0.0014849, 0.0011062
5: 0.0118266, 0.0207648, 0.0119805, 0.0241942, -0.0123676, 0.0087843
6: -0.0026432, -0.0014609, -0.0027491, -0.0015000, -0.0011432, 0.0012882
7: -0.0099764, -0.0069174, -0.0102504, -0.0070185, -0.0029579, 0.0033330
8: -0.0048106, -0.0026352, -0.0049547, -0.0012483, -0.0035623, 0.0023195
9: 0.0018489, 0.0037143, 0.0019106, 0.0038814, -0.0020325, 0.0018037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067246, upper bound: 0.0066588
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067165, upper bound: 0.0067061
time: 1.03 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9768384, 0.9896186, 0.9811447, 0.9897113, -0.0128729, 0.0084739
1: -0.0045891, -0.0038401, -0.0045265, -0.0038276, -0.0007615, 0.0006864
2: 0.0103528, 0.0142659, 0.0102304, 0.0139341, -0.0035813, 0.0040355
3: -0.0079290, -0.0059853, -0.0077108, -0.0059296, -0.0019995, 0.0017256
4: 0.0025316, 0.0039928, 0.0025080, 0.0036379, -0.0011062, 0.0014849
5: 0.0119805, 0.0241942, 0.0118266, 0.0207648, -0.0087843, 0.0123676
6: -0.0027491, -0.0015000, -0.0026432, -0.0014609, -0.0012882, 0.0011432
7: -0.0102504, -0.0070185, -0.0099764, -0.0069174, -0.0033330, 0.0029579
8: -0.0049547, -0.0012483, -0.0048106, -0.0026352, -0.0023195, 0.0035623
9: 0.0019106, 0.0038814, 0.0018489, 0.0037143, -0.0018037, 0.0020325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066992, upper bound: 0.0066635
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066907, upper bound: 0.0067050
time: 0.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9768384, 0.9896186, 0.9768384, 0.9896186, -0.0127802, 0.0127802
1: -0.0045891, -0.0038401, -0.0045891, -0.0038401, -0.0007490, 0.0007490
2: 0.0103528, 0.0142659, 0.0103528, 0.0142659, -0.0039131, 0.0039131
3: -0.0079290, -0.0059853, -0.0079290, -0.0059853, -0.0019438, 0.0019438
4: 0.0025316, 0.0039928, 0.0025316, 0.0039928, -0.0014612, 0.0014612
5: 0.0119805, 0.0241942, 0.0119805, 0.0241942, -0.0122137, 0.0122137
6: -0.0027491, -0.0015000, -0.0027491, -0.0015000, -0.0012492, 0.0012492
7: -0.0102504, -0.0070185, -0.0102504, -0.0070185, -0.0032319, 0.0032319
8: -0.0049547, -0.0012483, -0.0049547, -0.0012483, -0.0037064, 0.0037064
9: 0.0019106, 0.0038814, 0.0019106, 0.0038814, -0.0019708, 0.0019708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066992, upper bound: 0.0066635
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066907, upper bound: 0.0067050
time: 0.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.08 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0067246, upper bound: 0.0066588
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0067165, upper bound: 0.0067061
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0067246, upper bound: 0.0066588
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0067165, upper bound: 0.0067061
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0066992, upper bound: 0.0066635
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0066907, upper bound: 0.0067050
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0066992, upper bound: 0.0066635
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -0.0066907, upper bound: 0.0067050

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9828985, 0.9894957, 0.9811471, 0.9896906, -0.0067921, 0.0083486
1: -0.0045010, -0.0038814, -0.0045265, -0.0038328, -0.0006682, 0.0006451
2: 0.0105151, 0.0137990, 0.0102578, 0.0139339, -0.0034188, 0.0035412
3: -0.0076219, -0.0060592, -0.0077107, -0.0059420, -0.0016799, 0.0016515
4: 0.0025631, 0.0034933, 0.0025133, 0.0036377, -0.0010746, 0.0009801
5: 0.0121847, 0.0193682, 0.0118610, 0.0207629, -0.0085782, 0.0075072
6: -0.0026001, -0.0015518, -0.0026431, -0.0014696, -0.0011304, 0.0010914
7: -0.0098648, -0.0071526, -0.0099762, -0.0069400, -0.0029248, 0.0028237
8: -0.0047520, -0.0032000, -0.0048106, -0.0026360, -0.0021160, 0.0016105
9: 0.0019924, 0.0036463, 0.0018627, 0.0037142, -0.0017219, 0.0017835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067219
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067219
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9811702, 0.9896011, 0.9811470, 0.9897001, -0.0085299, 0.0084541
1: -0.0045261, -0.0038551, -0.0045265, -0.0038304, -0.0006957, 0.0006714
2: 0.0103758, 0.0139321, 0.0102452, 0.0139339, -0.0035581, 0.0036869
3: -0.0077095, -0.0059958, -0.0077107, -0.0059363, -0.0017732, 0.0017150
4: 0.0025361, 0.0036358, 0.0025108, 0.0036377, -0.0011016, 0.0011250
5: 0.0120095, 0.0207445, 0.0118452, 0.0207630, -0.0087534, 0.0088993
6: -0.0026426, -0.0015073, -0.0026431, -0.0014656, -0.0011770, 0.0011358
7: -0.0099748, -0.0070375, -0.0099762, -0.0069296, -0.0030452, 0.0029387
8: -0.0048098, -0.0026434, -0.0048106, -0.0026360, -0.0021738, 0.0021671
9: 0.0019222, 0.0037133, 0.0018564, 0.0037142, -0.0017920, 0.0018569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067680
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067680
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9828985, 0.9894957, 0.9768413, 0.9895981, -0.0066996, 0.0126544
1: -0.0045010, -0.0038814, -0.0045891, -0.0038403, -0.0006607, 0.0007077
2: 0.0105151, 0.0137990, 0.0103799, 0.0142656, -0.0037505, 0.0034190
3: -0.0076219, -0.0060592, -0.0079289, -0.0059976, -0.0016243, 0.0018697
4: 0.0025631, 0.0034933, 0.0025369, 0.0039926, -0.0014295, 0.0009564
5: 0.0121847, 0.0193682, 0.0120147, 0.0241920, -0.0120072, 0.0073535
6: -0.0026001, -0.0015518, -0.0027490, -0.0015086, -0.0010914, 0.0011973
7: -0.0098648, -0.0071526, -0.0102502, -0.0070409, -0.0028239, 0.0030977
8: -0.0047520, -0.0032000, -0.0049547, -0.0012492, -0.0035027, 0.0017546
9: 0.0019924, 0.0036463, 0.0019243, 0.0038813, -0.0018889, 0.0017220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0066588
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0066588
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9811702, 0.9896011, 0.9768409, 0.9896075, -0.0084373, 0.0127603
1: -0.0045261, -0.0038551, -0.0045891, -0.0038403, -0.0006858, 0.0007340
2: 0.0103758, 0.0139321, 0.0103675, 0.0142657, -0.0038898, 0.0035646
3: -0.0077095, -0.0059958, -0.0079289, -0.0059920, -0.0017176, 0.0019332
4: 0.0025361, 0.0036358, 0.0025345, 0.0039926, -0.0014565, 0.0011013
5: 0.0120095, 0.0207445, 0.0119991, 0.0241923, -0.0121827, 0.0087455
6: -0.0026426, -0.0015073, -0.0027491, -0.0015047, -0.0011379, 0.0012417
7: -0.0099748, -0.0070375, -0.0102503, -0.0070306, -0.0029441, 0.0032127
8: -0.0048098, -0.0026434, -0.0049547, -0.0012491, -0.0035607, 0.0023112
9: 0.0019222, 0.0037133, 0.0019180, 0.0038813, -0.0019591, 0.0017953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0067061
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0067061
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9785030, 0.9894070, 0.9811471, 0.9896906, -0.0111876, 0.0082600
1: -0.0045649, -0.0039035, -0.0045265, -0.0038328, -0.0007321, 0.0006230
2: 0.0106323, 0.0141376, 0.0102578, 0.0139339, -0.0033016, 0.0038799
3: -0.0078447, -0.0061125, -0.0077107, -0.0059420, -0.0019027, 0.0015982
4: 0.0025857, 0.0038556, 0.0025133, 0.0036377, -0.0010520, 0.0013424
5: 0.0123321, 0.0228686, 0.0118610, 0.0207629, -0.0084309, 0.0110076
6: -0.0027082, -0.0015892, -0.0026431, -0.0014696, -0.0012385, 0.0010540
7: -0.0101445, -0.0072493, -0.0099762, -0.0069400, -0.0032045, 0.0027269
8: -0.0048990, -0.0017844, -0.0048106, -0.0026360, -0.0022631, 0.0030261
9: 0.0020514, 0.0038168, 0.0018627, 0.0037142, -0.0016629, 0.0019541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0066773
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0066773
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9768653, 0.9895065, 0.9811470, 0.9897001, -0.0128348, 0.0083596
1: -0.0045887, -0.0038417, -0.0045265, -0.0038304, -0.0007583, 0.0006848
2: 0.0105007, 0.0142638, 0.0102452, 0.0139339, -0.0034332, 0.0040186
3: -0.0079277, -0.0060526, -0.0077107, -0.0059363, -0.0019914, 0.0016581
4: 0.0025603, 0.0039906, 0.0025108, 0.0036377, -0.0010774, 0.0014798
5: 0.0121666, 0.0241728, 0.0118452, 0.0207630, -0.0085964, 0.0123276
6: -0.0027485, -0.0015133, -0.0026431, -0.0014656, -0.0012828, 0.0011298
7: -0.0102487, -0.0071406, -0.0099762, -0.0069296, -0.0033191, 0.0028356
8: -0.0049538, -0.0012570, -0.0048106, -0.0026360, -0.0023179, 0.0035536
9: 0.0019851, 0.0038804, 0.0018564, 0.0037142, -0.0017291, 0.0020240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0067200
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0067200
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9785030, 0.9894070, 0.9768413, 0.9895981, -0.0110951, 0.0125657
1: -0.0045649, -0.0039035, -0.0045891, -0.0038403, -0.0007246, 0.0006856
2: 0.0106323, 0.0141376, 0.0103799, 0.0142656, -0.0036334, 0.0037577
3: -0.0078447, -0.0061125, -0.0079289, -0.0059976, -0.0018471, 0.0018164
4: 0.0025857, 0.0038556, 0.0025369, 0.0039926, -0.0014069, 0.0013187
5: 0.0123321, 0.0228686, 0.0120147, 0.0241920, -0.0118599, 0.0108539
6: -0.0027082, -0.0015892, -0.0027490, -0.0015086, -0.0011996, 0.0011599
7: -0.0101445, -0.0072493, -0.0102502, -0.0070409, -0.0031036, 0.0030009
8: -0.0048990, -0.0017844, -0.0049547, -0.0012492, -0.0036498, 0.0031702
9: 0.0020514, 0.0038168, 0.0019243, 0.0038813, -0.0018300, 0.0018926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0066635
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0066635
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9768653, 0.9895065, 0.9768409, 0.9896075, -0.0127422, 0.0126657
1: -0.0045887, -0.0038417, -0.0045891, -0.0038403, -0.0007484, 0.0007474
2: 0.0105007, 0.0142638, 0.0103675, 0.0142657, -0.0037650, 0.0038963
3: -0.0079277, -0.0060526, -0.0079289, -0.0059920, -0.0019357, 0.0018763
4: 0.0025603, 0.0039906, 0.0025345, 0.0039926, -0.0014323, 0.0014561
5: 0.0121666, 0.0241728, 0.0119991, 0.0241923, -0.0120257, 0.0121737
6: -0.0027485, -0.0015133, -0.0027491, -0.0015047, -0.0012438, 0.0012357
7: -0.0102487, -0.0071406, -0.0102503, -0.0070306, -0.0032181, 0.0031096
8: -0.0049538, -0.0012570, -0.0049547, -0.0012491, -0.0037047, 0.0036977
9: 0.0019851, 0.0038804, 0.0019180, 0.0038813, -0.0018962, 0.0019624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0067050
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0067050
time: 1.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.61 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067219
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067219
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067680
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0067219, upper bound: 0.0067680
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0066588
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0066588
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0067061
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066773, upper bound: 0.0067061
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0066773
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0066773
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0067200
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066588, upper bound: 0.0067200
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0066635
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0066635
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0067050
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -0.0066464, upper bound: 0.0067050

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9828985, 0.9894957, 0.9828985, 0.9894957, -0.0065972, 0.0065972
1: -0.0045010, -0.0038814, -0.0045010, -0.0038814, -0.0006196, 0.0006196
2: 0.0105151, 0.0137990, 0.0105151, 0.0137990, -0.0032838, 0.0032838
3: -0.0076219, -0.0060592, -0.0076219, -0.0060592, -0.0015628, 0.0015628
4: 0.0025631, 0.0034933, 0.0025631, 0.0034933, -0.0009303, 0.0009303
5: 0.0121847, 0.0193682, 0.0121847, 0.0193682, -0.0071834, 0.0071834
6: -0.0026001, -0.0015518, -0.0026001, -0.0015518, -0.0010483, 0.0010483
7: -0.0098648, -0.0071526, -0.0098648, -0.0071526, -0.0027122, 0.0027122
8: -0.0047520, -0.0032000, -0.0047520, -0.0032000, -0.0015519, 0.0015519
9: 0.0019924, 0.0036463, 0.0019924, 0.0036463, -0.0016539, 0.0016539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0065653
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0066424
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9828985, 0.9894957, 0.9811702, 0.9896011, -0.0067027, 0.0083255
1: -0.0045010, -0.0038814, -0.0045261, -0.0038551, -0.0006459, 0.0006448
2: 0.0105151, 0.0137990, 0.0103758, 0.0139321, -0.0034170, 0.0034231
3: -0.0076219, -0.0060592, -0.0077095, -0.0059958, -0.0016262, 0.0016504
4: 0.0025631, 0.0034933, 0.0025361, 0.0036358, -0.0010727, 0.0009572
5: 0.0121847, 0.0193682, 0.0120095, 0.0207445, -0.0085598, 0.0073586
6: -0.0026001, -0.0015518, -0.0026426, -0.0015073, -0.0010928, 0.0010908
7: -0.0098648, -0.0071526, -0.0099748, -0.0070375, -0.0028273, 0.0028222
8: -0.0047520, -0.0032000, -0.0048098, -0.0026434, -0.0021085, 0.0016097
9: 0.0019924, 0.0036463, 0.0019222, 0.0037133, -0.0017210, 0.0017241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0065653
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0066424
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9811702, 0.9896011, 0.9828985, 0.9894957, -0.0083255, 0.0067027
1: -0.0045261, -0.0038551, -0.0045010, -0.0038814, -0.0006448, 0.0006459
2: 0.0103758, 0.0139321, 0.0105151, 0.0137990, -0.0034231, 0.0034170
3: -0.0077095, -0.0059958, -0.0076219, -0.0060592, -0.0016504, 0.0016262
4: 0.0025361, 0.0036358, 0.0025631, 0.0034933, -0.0009572, 0.0010727
5: 0.0120095, 0.0207445, 0.0121847, 0.0193682, -0.0073586, 0.0085598
6: -0.0026426, -0.0015073, -0.0026001, -0.0015518, -0.0010908, 0.0010928
7: -0.0099748, -0.0070375, -0.0098648, -0.0071526, -0.0028222, 0.0028273
8: -0.0048098, -0.0026434, -0.0047520, -0.0032000, -0.0016097, 0.0021085
9: 0.0019222, 0.0037133, 0.0019924, 0.0036463, -0.0017241, 0.0017210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066063
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066916
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9811702, 0.9896011, 0.9811702, 0.9896011, -0.0084310, 0.0084310
1: -0.0045261, -0.0038551, -0.0045261, -0.0038551, -0.0006711, 0.0006711
2: 0.0103758, 0.0139321, 0.0103758, 0.0139321, -0.0035563, 0.0035563
3: -0.0077095, -0.0059958, -0.0077095, -0.0059958, -0.0017138, 0.0017138
4: 0.0025361, 0.0036358, 0.0025361, 0.0036358, -0.0010997, 0.0010997
5: 0.0120095, 0.0207445, 0.0120095, 0.0207445, -0.0087350, 0.0087350
6: -0.0026426, -0.0015073, -0.0026426, -0.0015073, -0.0011353, 0.0011353
7: -0.0099748, -0.0070375, -0.0099748, -0.0070375, -0.0029373, 0.0029373
8: -0.0048098, -0.0026434, -0.0048098, -0.0026434, -0.0021664, 0.0021664
9: 0.0019222, 0.0037133, 0.0019222, 0.0037133, -0.0017911, 0.0017911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066063
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066916
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9828985, 0.9894957, 0.9785030, 0.9894070, -0.0065085, 0.0109927
1: -0.0045010, -0.0038814, -0.0045649, -0.0039035, -0.0005975, 0.0006836
2: 0.0105151, 0.0137990, 0.0106323, 0.0141376, -0.0036225, 0.0031667
3: -0.0076219, -0.0060592, -0.0078447, -0.0061125, -0.0015095, 0.0017855
4: 0.0025631, 0.0034933, 0.0025857, 0.0038556, -0.0012926, 0.0009076
5: 0.0121847, 0.0193682, 0.0123321, 0.0228686, -0.0106839, 0.0070361
6: -0.0026001, -0.0015518, -0.0027082, -0.0015892, -0.0010109, 0.0011564
7: -0.0098648, -0.0071526, -0.0101445, -0.0072493, -0.0026155, 0.0029919
8: -0.0047520, -0.0032000, -0.0048990, -0.0017844, -0.0029675, 0.0016990
9: 0.0019924, 0.0036463, 0.0020514, 0.0038168, -0.0018245, 0.0015949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0064983
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0065788
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9828985, 0.9894957, 0.9768653, 0.9895065, -0.0066081, 0.0126304
1: -0.0045010, -0.0038814, -0.0045887, -0.0038417, -0.0006593, 0.0007074
2: 0.0105151, 0.0137990, 0.0105007, 0.0142638, -0.0037486, 0.0032983
3: -0.0076219, -0.0060592, -0.0079277, -0.0060526, -0.0015694, 0.0018685
4: 0.0025631, 0.0034933, 0.0025603, 0.0039906, -0.0014275, 0.0009331
5: 0.0121847, 0.0193682, 0.0121666, 0.0241728, -0.0119881, 0.0072016
6: -0.0026001, -0.0015518, -0.0027485, -0.0015133, -0.0010867, 0.0011967
7: -0.0098648, -0.0071526, -0.0102487, -0.0071406, -0.0027242, 0.0030961
8: -0.0047520, -0.0032000, -0.0049538, -0.0012570, -0.0034950, 0.0017538
9: 0.0019924, 0.0036463, 0.0019851, 0.0038804, -0.0018880, 0.0016612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0064983
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0065788
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9811702, 0.9896011, 0.9785030, 0.9894070, -0.0082368, 0.0110981
1: -0.0045261, -0.0038551, -0.0045649, -0.0039035, -0.0006227, 0.0007098
2: 0.0103758, 0.0139321, 0.0106323, 0.0141376, -0.0037618, 0.0032999
3: -0.0077095, -0.0059958, -0.0078447, -0.0061125, -0.0015971, 0.0018489
4: 0.0025361, 0.0036358, 0.0025857, 0.0038556, -0.0013195, 0.0010501
5: 0.0120095, 0.0207445, 0.0123321, 0.0228686, -0.0108591, 0.0084125
6: -0.0026426, -0.0015073, -0.0027082, -0.0015892, -0.0010534, 0.0012009
7: -0.0099748, -0.0070375, -0.0101445, -0.0072493, -0.0027255, 0.0031070
8: -0.0048098, -0.0026434, -0.0048990, -0.0017844, -0.0030254, 0.0022556
9: 0.0019222, 0.0037133, 0.0020514, 0.0038168, -0.0018946, 0.0016620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0065396
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0066296
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9811702, 0.9896011, 0.9768653, 0.9895065, -0.0083364, 0.0127358
1: -0.0045261, -0.0038551, -0.0045887, -0.0038417, -0.0006844, 0.0007336
2: 0.0103758, 0.0139321, 0.0105007, 0.0142638, -0.0038879, 0.0034314
3: -0.0077095, -0.0059958, -0.0079277, -0.0060526, -0.0016569, 0.0019319
4: 0.0025361, 0.0036358, 0.0025603, 0.0039906, -0.0014545, 0.0010755
5: 0.0120095, 0.0207445, 0.0121666, 0.0241728, -0.0121633, 0.0085780
6: -0.0026426, -0.0015073, -0.0027485, -0.0015133, -0.0011292, 0.0012411
7: -0.0099748, -0.0070375, -0.0102487, -0.0071406, -0.0028341, 0.0032112
8: -0.0048098, -0.0026434, -0.0049538, -0.0012570, -0.0035528, 0.0023104
9: 0.0019222, 0.0037133, 0.0019851, 0.0038804, -0.0019582, 0.0017282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0065396
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0066296
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9785030, 0.9894070, 0.9828985, 0.9894957, -0.0109927, 0.0065085
1: -0.0045649, -0.0039035, -0.0045010, -0.0038814, -0.0006836, 0.0005975
2: 0.0106323, 0.0141376, 0.0105151, 0.0137990, -0.0031667, 0.0036225
3: -0.0078447, -0.0061125, -0.0076219, -0.0060592, -0.0017855, 0.0015095
4: 0.0025857, 0.0038556, 0.0025631, 0.0034933, -0.0009076, 0.0012926
5: 0.0123321, 0.0228686, 0.0121847, 0.0193682, -0.0070361, 0.0106839
6: -0.0027082, -0.0015892, -0.0026001, -0.0015518, -0.0011564, 0.0010109
7: -0.0101445, -0.0072493, -0.0098648, -0.0071526, -0.0029919, 0.0026155
8: -0.0048990, -0.0017844, -0.0047520, -0.0032000, -0.0016990, 0.0029675
9: 0.0020514, 0.0038168, 0.0019924, 0.0036463, -0.0015949, 0.0018245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065166
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065995
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9785030, 0.9894070, 0.9811702, 0.9896011, -0.0110981, 0.0082368
1: -0.0045649, -0.0039035, -0.0045261, -0.0038551, -0.0007098, 0.0006227
2: 0.0106323, 0.0141376, 0.0103758, 0.0139321, -0.0032999, 0.0037618
3: -0.0078447, -0.0061125, -0.0077095, -0.0059958, -0.0018489, 0.0015971
4: 0.0025857, 0.0038556, 0.0025361, 0.0036358, -0.0010501, 0.0013195
5: 0.0123321, 0.0228686, 0.0120095, 0.0207445, -0.0084125, 0.0108591
6: -0.0027082, -0.0015892, -0.0026426, -0.0015073, -0.0012009, 0.0010534
7: -0.0101445, -0.0072493, -0.0099748, -0.0070375, -0.0031070, 0.0027255
8: -0.0048990, -0.0017844, -0.0048098, -0.0026434, -0.0022556, 0.0030254
9: 0.0020514, 0.0038168, 0.0019222, 0.0037133, -0.0016620, 0.0018946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065166
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065995
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9768653, 0.9895065, 0.9828985, 0.9894957, -0.0126304, 0.0066081
1: -0.0045887, -0.0038417, -0.0045010, -0.0038814, -0.0007074, 0.0006593
2: 0.0105007, 0.0142638, 0.0105151, 0.0137990, -0.0032983, 0.0037486
3: -0.0079277, -0.0060526, -0.0076219, -0.0060592, -0.0018685, 0.0015694
4: 0.0025603, 0.0039906, 0.0025631, 0.0034933, -0.0009331, 0.0014275
5: 0.0121666, 0.0241728, 0.0121847, 0.0193682, -0.0072016, 0.0119881
6: -0.0027485, -0.0015133, -0.0026001, -0.0015518, -0.0011967, 0.0010867
7: -0.0102487, -0.0071406, -0.0098648, -0.0071526, -0.0030961, 0.0027242
8: -0.0049538, -0.0012570, -0.0047520, -0.0032000, -0.0017538, 0.0034950
9: 0.0019851, 0.0038804, 0.0019924, 0.0036463, -0.0016612, 0.0018880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0065518
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0066458
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9768653, 0.9895065, 0.9811702, 0.9896011, -0.0127358, 0.0083364
1: -0.0045887, -0.0038417, -0.0045261, -0.0038551, -0.0007336, 0.0006844
2: 0.0105007, 0.0142638, 0.0103758, 0.0139321, -0.0034314, 0.0038879
3: -0.0079277, -0.0060526, -0.0077095, -0.0059958, -0.0019319, 0.0016569
4: 0.0025603, 0.0039906, 0.0025361, 0.0036358, -0.0010755, 0.0014545
5: 0.0121666, 0.0241728, 0.0120095, 0.0207445, -0.0085780, 0.0121633
6: -0.0027485, -0.0015133, -0.0026426, -0.0015073, -0.0012411, 0.0011292
7: -0.0102487, -0.0071406, -0.0099748, -0.0070375, -0.0032112, 0.0028341
8: -0.0049538, -0.0012570, -0.0048098, -0.0026434, -0.0023104, 0.0035528
9: 0.0019851, 0.0038804, 0.0019222, 0.0037133, -0.0017282, 0.0019582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0065518
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0066458
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9785030, 0.9894070, 0.9785030, 0.9894070, -0.0109040, 0.0109040
1: -0.0045649, -0.0039035, -0.0045649, -0.0039035, -0.0006614, 0.0006614
2: 0.0106323, 0.0141376, 0.0106323, 0.0141376, -0.0035053, 0.0035053
3: -0.0078447, -0.0061125, -0.0078447, -0.0061125, -0.0017322, 0.0017322
4: 0.0025857, 0.0038556, 0.0025857, 0.0038556, -0.0012699, 0.0012699
5: 0.0123321, 0.0228686, 0.0123321, 0.0228686, -0.0105366, 0.0105366
6: -0.0027082, -0.0015892, -0.0027082, -0.0015892, -0.0011190, 0.0011190
7: -0.0101445, -0.0072493, -0.0101445, -0.0072493, -0.0028952, 0.0028952
8: -0.0048990, -0.0017844, -0.0048990, -0.0017844, -0.0031146, 0.0031146
9: 0.0020514, 0.0038168, 0.0020514, 0.0038168, -0.0017655, 0.0017655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0064981
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0065848
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9785030, 0.9894070, 0.9768653, 0.9895065, -0.0110036, 0.0125417
1: -0.0045649, -0.0039035, -0.0045887, -0.0038417, -0.0007232, 0.0006852
2: 0.0106323, 0.0141376, 0.0105007, 0.0142638, -0.0036315, 0.0036369
3: -0.0078447, -0.0061125, -0.0079277, -0.0060526, -0.0017921, 0.0018152
4: 0.0025857, 0.0038556, 0.0025603, 0.0039906, -0.0014049, 0.0012954
5: 0.0123321, 0.0228686, 0.0121666, 0.0241728, -0.0118407, 0.0107020
6: -0.0027082, -0.0015892, -0.0027485, -0.0015133, -0.0011948, 0.0011593
7: -0.0101445, -0.0072493, -0.0102487, -0.0071406, -0.0030039, 0.0029994
8: -0.0048990, -0.0017844, -0.0049538, -0.0012570, -0.0036421, 0.0031694
9: 0.0020514, 0.0038168, 0.0019851, 0.0038804, -0.0018290, 0.0018317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0064981
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0065848
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9768653, 0.9895065, 0.9785030, 0.9894070, -0.0125417, 0.0110036
1: -0.0045887, -0.0038417, -0.0045649, -0.0039035, -0.0006852, 0.0007232
2: 0.0105007, 0.0142638, 0.0106323, 0.0141376, -0.0036369, 0.0036315
3: -0.0079277, -0.0060526, -0.0078447, -0.0061125, -0.0018152, 0.0017921
4: 0.0025603, 0.0039906, 0.0025857, 0.0038556, -0.0012954, 0.0014049
5: 0.0121666, 0.0241728, 0.0123321, 0.0228686, -0.0107020, 0.0118407
6: -0.0027485, -0.0015133, -0.0027082, -0.0015892, -0.0011593, 0.0011948
7: -0.0102487, -0.0071406, -0.0101445, -0.0072493, -0.0029994, 0.0030039
8: -0.0049538, -0.0012570, -0.0048990, -0.0017844, -0.0031694, 0.0036421
9: 0.0019851, 0.0038804, 0.0020514, 0.0038168, -0.0018317, 0.0018290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0065330
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0066300
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9768653, 0.9895065, 0.9768653, 0.9895065, -0.0126413, 0.0126413
1: -0.0045887, -0.0038417, -0.0045887, -0.0038417, -0.0007470, 0.0007470
2: 0.0105007, 0.0142638, 0.0105007, 0.0142638, -0.0037631, 0.0037631
3: -0.0079277, -0.0060526, -0.0079277, -0.0060526, -0.0018751, 0.0018751
4: 0.0025603, 0.0039906, 0.0025603, 0.0039906, -0.0014303, 0.0014303
5: 0.0121666, 0.0241728, 0.0121666, 0.0241728, -0.0120062, 0.0120062
6: -0.0027485, -0.0015133, -0.0027485, -0.0015133, -0.0012351, 0.0012351
7: -0.0102487, -0.0071406, -0.0102487, -0.0071406, -0.0031081, 0.0031081
8: -0.0049538, -0.0012570, -0.0049538, -0.0012570, -0.0036969, 0.0036969
9: 0.0019851, 0.0038804, 0.0019851, 0.0038804, -0.0018953, 0.0018953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0065330
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0066300
time: 1.08 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.26 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0065653
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0066424
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0065653
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066824, upper bound: 0.0066424
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066063
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066916
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066063
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066424, upper bound: 0.0066916
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0064983
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0065788
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0064983
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066372, upper bound: 0.0065788
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0065396
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0066296
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0065396
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065995, upper bound: 0.0066296
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065166
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065995
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065166
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066187, upper bound: 0.0065995
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0065518
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0066458
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0065518
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065788, upper bound: 0.0066458
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0064981
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0065848
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0064981
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0066071, upper bound: 0.0065848
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0065330
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0066300
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0065330
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -0.0065691, upper bound: 0.0066300

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9829116, 0.9894547, -0.0056384, 0.0063393
1: -0.0044877, -0.0039424, -0.0045008, -0.0038916, -0.0005961, 0.0005585
2: 0.0108384, 0.0137283, 0.0105692, 0.0137980, -0.0029595, 0.0031591
3: -0.0075754, -0.0062063, -0.0076213, -0.0060838, -0.0014917, 0.0014150
4: 0.0026256, 0.0034177, 0.0025735, 0.0034922, -0.0008666, 0.0008442
5: 0.0125913, 0.0186372, 0.0122527, 0.0193577, -0.0067663, 0.0063844
6: -0.0025775, -0.0016550, -0.0025997, -0.0015690, -0.0010085, 0.0009448
7: -0.0098064, -0.0074196, -0.0098640, -0.0071972, -0.0026092, 0.0024444
8: -0.0047212, -0.0034660, -0.0047515, -0.0032043, -0.0015169, 0.0012855
9: 0.0021552, 0.0036106, 0.0020196, 0.0036458, -0.0014906, 0.0015911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066030
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066030
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9828985, 0.9894956, -0.0065647, 0.0065199
1: -0.0045005, -0.0039006, -0.0045010, -0.0038814, -0.0006191, 0.0006004
2: 0.0106171, 0.0137965, 0.0105153, 0.0137990, -0.0031818, 0.0032811
3: -0.0076203, -0.0061056, -0.0076220, -0.0060592, -0.0015611, 0.0015164
4: 0.0025828, 0.0034907, 0.0025631, 0.0034933, -0.0009105, 0.0009276
5: 0.0123130, 0.0193423, 0.0121850, 0.0193681, -0.0070551, 0.0071574
6: -0.0025993, -0.0015843, -0.0026001, -0.0015519, -0.0010474, 0.0010157
7: -0.0098627, -0.0072368, -0.0098648, -0.0071527, -0.0027100, 0.0026280
8: -0.0047509, -0.0032105, -0.0047519, -0.0032001, -0.0015508, 0.0015414
9: 0.0020437, 0.0036450, 0.0019924, 0.0036463, -0.0016025, 0.0016526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066834
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066834
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9811828, 0.9895594, -0.0057430, 0.0080681
1: -0.0044877, -0.0039424, -0.0045259, -0.0038655, -0.0006222, 0.0005836
2: 0.0108384, 0.0137283, 0.0104310, 0.0139312, -0.0030927, 0.0032972
3: -0.0075754, -0.0062063, -0.0077089, -0.0060209, -0.0015546, 0.0015026
4: 0.0026256, 0.0034177, 0.0025468, 0.0036348, -0.0010091, 0.0008709
5: 0.0125913, 0.0186372, 0.0120789, 0.0207345, -0.0081432, 0.0065582
6: -0.0025775, -0.0016550, -0.0026423, -0.0015249, -0.0010526, 0.0009873
7: -0.0098064, -0.0074196, -0.0099740, -0.0070831, -0.0027233, 0.0025544
8: -0.0047212, -0.0034660, -0.0048094, -0.0026475, -0.0020738, 0.0013433
9: 0.0021552, 0.0036106, 0.0019500, 0.0037128, -0.0015577, 0.0016607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0065653
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0065653
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9811702, 0.9896010, -0.0066701, 0.0082483
1: -0.0045005, -0.0039006, -0.0045261, -0.0038551, -0.0006454, 0.0006255
2: 0.0106171, 0.0137965, 0.0103760, 0.0139321, -0.0033150, 0.0034204
3: -0.0076203, -0.0061056, -0.0077095, -0.0059958, -0.0016245, 0.0016040
4: 0.0025828, 0.0034907, 0.0025361, 0.0036358, -0.0010530, 0.0009545
5: 0.0123130, 0.0193423, 0.0120098, 0.0207445, -0.0084315, 0.0073326
6: -0.0025993, -0.0015843, -0.0026426, -0.0015074, -0.0010919, 0.0010582
7: -0.0098627, -0.0072368, -0.0099748, -0.0070377, -0.0028251, 0.0027380
8: -0.0047509, -0.0032105, -0.0048098, -0.0026434, -0.0021074, 0.0015993
9: 0.0020437, 0.0036450, 0.0019223, 0.0037133, -0.0016696, 0.0017227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0066424
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0066424
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9829116, 0.9894547, -0.0071691, 0.0064337
1: -0.0045099, -0.0039188, -0.0045008, -0.0038916, -0.0006183, 0.0005820
2: 0.0107136, 0.0138462, 0.0105692, 0.0137980, -0.0030843, 0.0032770
3: -0.0076530, -0.0061495, -0.0076213, -0.0060838, -0.0015692, 0.0014718
4: 0.0026015, 0.0035439, 0.0025735, 0.0034922, -0.0008908, 0.0009703
5: 0.0124344, 0.0198562, 0.0122527, 0.0193577, -0.0069233, 0.0076035
6: -0.0026151, -0.0016151, -0.0025997, -0.0015690, -0.0010461, 0.0009846
7: -0.0099038, -0.0073165, -0.0098640, -0.0071972, -0.0027066, 0.0025475
8: -0.0047725, -0.0030027, -0.0047515, -0.0032043, -0.0015682, 0.0017488
9: 0.0020923, 0.0036700, 0.0020196, 0.0036458, -0.0015534, 0.0016505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066144
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066144
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9828985, 0.9894956, -0.0082963, 0.0066313
1: -0.0045257, -0.0038729, -0.0045010, -0.0038814, -0.0006443, 0.0006281
2: 0.0104701, 0.0139299, 0.0105153, 0.0137990, -0.0033289, 0.0034146
3: -0.0077081, -0.0060387, -0.0076220, -0.0060592, -0.0016488, 0.0015833
4: 0.0025544, 0.0036334, 0.0025631, 0.0034933, -0.0009390, 0.0010703
5: 0.0121281, 0.0207213, 0.0121850, 0.0193681, -0.0072400, 0.0085364
6: -0.0026419, -0.0015374, -0.0026001, -0.0015519, -0.0010900, 0.0010627
7: -0.0099729, -0.0071154, -0.0098648, -0.0071527, -0.0028202, 0.0027494
8: -0.0048088, -0.0026528, -0.0047519, -0.0032001, -0.0016087, 0.0020991
9: 0.0019697, 0.0037122, 0.0019924, 0.0036463, -0.0016766, 0.0017198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0067005
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0067005
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9811828, 0.9895594, -0.0072737, 0.0081626
1: -0.0045099, -0.0039188, -0.0045259, -0.0038655, -0.0006444, 0.0006071
2: 0.0107136, 0.0138462, 0.0104310, 0.0139312, -0.0032175, 0.0034152
3: -0.0076530, -0.0061495, -0.0077089, -0.0060209, -0.0016321, 0.0015594
4: 0.0026015, 0.0035439, 0.0025468, 0.0036348, -0.0010333, 0.0009971
5: 0.0124344, 0.0198562, 0.0120789, 0.0207345, -0.0083001, 0.0077773
6: -0.0026151, -0.0016151, -0.0026423, -0.0015249, -0.0010902, 0.0010271
7: -0.0099038, -0.0073165, -0.0099740, -0.0070831, -0.0028207, 0.0026575
8: -0.0047725, -0.0030027, -0.0048094, -0.0026475, -0.0021250, 0.0018067
9: 0.0020923, 0.0036700, 0.0019500, 0.0037128, -0.0016205, 0.0017201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066063
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066063
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9811702, 0.9896010, -0.0084018, 0.0083596
1: -0.0045257, -0.0038729, -0.0045261, -0.0038551, -0.0006706, 0.0006533
2: 0.0104701, 0.0139299, 0.0103760, 0.0139321, -0.0034620, 0.0035539
3: -0.0077081, -0.0060387, -0.0077095, -0.0059958, -0.0017122, 0.0016709
4: 0.0025544, 0.0036334, 0.0025361, 0.0036358, -0.0010814, 0.0010972
5: 0.0121281, 0.0207213, 0.0120098, 0.0207445, -0.0086164, 0.0087116
6: -0.0026419, -0.0015374, -0.0026426, -0.0015074, -0.0011345, 0.0011052
7: -0.0099729, -0.0071154, -0.0099748, -0.0070377, -0.0029353, 0.0028594
8: -0.0048088, -0.0026528, -0.0048098, -0.0026434, -0.0021654, 0.0021570
9: 0.0019697, 0.0037122, 0.0019223, 0.0037133, -0.0017436, 0.0017899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066916
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066916
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9785167, 0.9893657, -0.0055493, 0.0107341
1: -0.0044877, -0.0039424, -0.0045647, -0.0039137, -0.0005739, 0.0006223
2: 0.0108384, 0.0137283, 0.0106867, 0.0141366, -0.0032982, 0.0030416
3: -0.0075754, -0.0062063, -0.0078440, -0.0061372, -0.0014382, 0.0016377
4: 0.0026256, 0.0034177, 0.0025963, 0.0038545, -0.0012289, 0.0008214
5: 0.0125913, 0.0186372, 0.0124005, 0.0228577, -0.0102664, 0.0062367
6: -0.0025775, -0.0016550, -0.0027078, -0.0016066, -0.0009709, 0.0010529
7: -0.0098064, -0.0074196, -0.0101436, -0.0072943, -0.0025121, 0.0027241
8: -0.0047212, -0.0034660, -0.0048986, -0.0017888, -0.0029324, 0.0014326
9: 0.0021552, 0.0036106, 0.0020788, 0.0038163, -0.0016611, 0.0015319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0065358
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0065358
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9785030, 0.9894069, -0.0064760, 0.0109155
1: -0.0045005, -0.0039006, -0.0045649, -0.0039035, -0.0005970, 0.0006643
2: 0.0106171, 0.0137965, 0.0106324, 0.0141376, -0.0035205, 0.0031640
3: -0.0076203, -0.0061056, -0.0078447, -0.0061126, -0.0015078, 0.0017391
4: 0.0025828, 0.0034907, 0.0025858, 0.0038556, -0.0012728, 0.0009049
5: 0.0123130, 0.0193423, 0.0123323, 0.0228685, -0.0105555, 0.0070101
6: -0.0025993, -0.0015843, -0.0027082, -0.0015892, -0.0010100, 0.0011238
7: -0.0098627, -0.0072368, -0.0101445, -0.0072494, -0.0026133, 0.0029077
8: -0.0047509, -0.0032105, -0.0048990, -0.0017844, -0.0029664, 0.0016885
9: 0.0020437, 0.0036450, 0.0020514, 0.0038168, -0.0017731, 0.0015936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0066207
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0066207
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9768788, 0.9894645, -0.0056481, 0.0123721
1: -0.0044877, -0.0039424, -0.0045885, -0.0038425, -0.0006452, 0.0006462
2: 0.0108384, 0.0137283, 0.0105563, 0.0142628, -0.0034244, 0.0031719
3: -0.0075754, -0.0062063, -0.0079270, -0.0060779, -0.0014975, 0.0017207
4: 0.0026256, 0.0034177, 0.0025710, 0.0039895, -0.0013639, 0.0008467
5: 0.0125913, 0.0186372, 0.0122365, 0.0241622, -0.0115709, 0.0064007
6: -0.0025775, -0.0016550, -0.0027481, -0.0015146, -0.0010629, 0.0010931
7: -0.0098064, -0.0074196, -0.0102479, -0.0071866, -0.0026198, 0.0028283
8: -0.0047212, -0.0034660, -0.0049534, -0.0012613, -0.0034600, 0.0014874
9: 0.0021552, 0.0036106, 0.0020131, 0.0038799, -0.0017247, 0.0015976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0064983
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0064983
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9768654, 0.9895065, -0.0065756, 0.0125531
1: -0.0045005, -0.0039006, -0.0045887, -0.0038417, -0.0006588, 0.0006881
2: 0.0106171, 0.0137965, 0.0105009, 0.0142638, -0.0036466, 0.0032956
3: -0.0076203, -0.0061056, -0.0079277, -0.0060527, -0.0015676, 0.0018221
4: 0.0025828, 0.0034907, 0.0025603, 0.0039906, -0.0014078, 0.0009304
5: 0.0123130, 0.0193423, 0.0121668, 0.0241728, -0.0118598, 0.0071755
6: -0.0025993, -0.0015843, -0.0027485, -0.0015134, -0.0010859, 0.0011641
7: -0.0098627, -0.0072368, -0.0102487, -0.0071408, -0.0027220, 0.0030119
8: -0.0047509, -0.0032105, -0.0049538, -0.0012570, -0.0034939, 0.0017433
9: 0.0020437, 0.0036450, 0.0019852, 0.0038804, -0.0018366, 0.0016598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0065788
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0065788
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9785167, 0.9893657, -0.0070800, 0.0108286
1: -0.0045099, -0.0039188, -0.0045647, -0.0039137, -0.0005962, 0.0006459
2: 0.0107136, 0.0138462, 0.0106867, 0.0141366, -0.0034229, 0.0031595
3: -0.0076530, -0.0061495, -0.0078440, -0.0061372, -0.0015158, 0.0016945
4: 0.0026015, 0.0035439, 0.0025963, 0.0038545, -0.0012530, 0.0009476
5: 0.0124344, 0.0198562, 0.0124005, 0.0228577, -0.0104233, 0.0074557
6: -0.0026151, -0.0016151, -0.0027078, -0.0016066, -0.0010086, 0.0010927
7: -0.0099038, -0.0073165, -0.0101436, -0.0072943, -0.0026095, 0.0028271
8: -0.0047725, -0.0030027, -0.0048986, -0.0017888, -0.0029836, 0.0018959
9: 0.0020923, 0.0036700, 0.0020788, 0.0038163, -0.0017240, 0.0015913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065469
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065469
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9785030, 0.9894069, -0.0082076, 0.0110268
1: -0.0045257, -0.0038729, -0.0045649, -0.0039035, -0.0006222, 0.0006921
2: 0.0104701, 0.0139299, 0.0106324, 0.0141376, -0.0036675, 0.0032974
3: -0.0077081, -0.0060387, -0.0078447, -0.0061126, -0.0015955, 0.0018060
4: 0.0025544, 0.0036334, 0.0025858, 0.0038556, -0.0013013, 0.0010476
5: 0.0121281, 0.0207213, 0.0123323, 0.0228685, -0.0107405, 0.0083891
6: -0.0026419, -0.0015374, -0.0027082, -0.0015892, -0.0010526, 0.0011708
7: -0.0099729, -0.0071154, -0.0101445, -0.0072494, -0.0027235, 0.0030291
8: -0.0048088, -0.0026528, -0.0048990, -0.0017844, -0.0030244, 0.0022462
9: 0.0019697, 0.0037122, 0.0020514, 0.0038168, -0.0018471, 0.0016608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066375
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066375
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9768788, 0.9894645, -0.0071788, 0.0124665
1: -0.0045099, -0.0039188, -0.0045885, -0.0038425, -0.0006675, 0.0006697
2: 0.0107136, 0.0138462, 0.0105563, 0.0142628, -0.0035491, 0.0032899
3: -0.0076530, -0.0061495, -0.0079270, -0.0060779, -0.0015751, 0.0017775
4: 0.0026015, 0.0035439, 0.0025710, 0.0039895, -0.0013880, 0.0009728
5: 0.0124344, 0.0198562, 0.0122365, 0.0241622, -0.0117278, 0.0076197
6: -0.0026151, -0.0016151, -0.0027481, -0.0015146, -0.0011005, 0.0011330
7: -0.0099038, -0.0073165, -0.0102479, -0.0071866, -0.0027172, 0.0029314
8: -0.0047725, -0.0030027, -0.0049534, -0.0012613, -0.0035112, 0.0019507
9: 0.0020923, 0.0036700, 0.0020131, 0.0038799, -0.0017875, 0.0016570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065396
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065396
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9768654, 0.9895065, -0.0083072, 0.0126644
1: -0.0045257, -0.0038729, -0.0045887, -0.0038417, -0.0006840, 0.0007159
2: 0.0104701, 0.0139299, 0.0105009, 0.0142638, -0.0037937, 0.0034290
3: -0.0077081, -0.0060387, -0.0079277, -0.0060527, -0.0016554, 0.0018890
4: 0.0025544, 0.0036334, 0.0025603, 0.0039906, -0.0014363, 0.0010731
5: 0.0121281, 0.0207213, 0.0121668, 0.0241728, -0.0120447, 0.0085545
6: -0.0026419, -0.0015374, -0.0027485, -0.0015134, -0.0011285, 0.0012110
7: -0.0099729, -0.0071154, -0.0102487, -0.0071408, -0.0028321, 0.0031333
8: -0.0048088, -0.0026528, -0.0049538, -0.0012570, -0.0035518, 0.0023010
9: 0.0019697, 0.0037122, 0.0019852, 0.0038804, -0.0019107, 0.0017270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066296
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066296
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9829116, 0.9894547, -0.0099178, 0.0062445
1: -0.0045499, -0.0039660, -0.0045008, -0.0038916, -0.0006583, 0.0005349
2: 0.0109635, 0.0140580, 0.0105692, 0.0137980, -0.0028344, 0.0034888
3: -0.0077923, -0.0062632, -0.0076213, -0.0060838, -0.0017085, 0.0013580
4: 0.0026499, 0.0037704, 0.0025735, 0.0034922, -0.0008424, 0.0011969
5: 0.0127487, 0.0220452, 0.0122527, 0.0193577, -0.0066090, 0.0097924
6: -0.0026827, -0.0016949, -0.0025997, -0.0015690, -0.0011137, 0.0009048
7: -0.0100787, -0.0075229, -0.0098640, -0.0071972, -0.0028815, 0.0023411
8: -0.0048644, -0.0021174, -0.0047515, -0.0032043, -0.0016601, 0.0026341
9: 0.0022182, 0.0037767, 0.0020196, 0.0036458, -0.0014276, 0.0017571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0065544
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0065544
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9828985, 0.9894956, -0.0109586, 0.0064351
1: -0.0045644, -0.0039217, -0.0045010, -0.0038814, -0.0006830, 0.0005793
2: 0.0107291, 0.0141350, 0.0105153, 0.0137990, -0.0030699, 0.0036197
3: -0.0078430, -0.0061565, -0.0076220, -0.0060592, -0.0017837, 0.0014654
4: 0.0026045, 0.0038528, 0.0025631, 0.0034933, -0.0008889, 0.0012897
5: 0.0124538, 0.0228415, 0.0121850, 0.0193681, -0.0069143, 0.0106565
6: -0.0027073, -0.0016201, -0.0026001, -0.0015519, -0.0011555, 0.0009800
7: -0.0101423, -0.0073292, -0.0098648, -0.0071527, -0.0029896, 0.0025356
8: -0.0048979, -0.0017954, -0.0047519, -0.0032001, -0.0016978, 0.0029566
9: 0.0021001, 0.0038155, 0.0019924, 0.0036463, -0.0015462, 0.0018231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0066396
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0066396
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9811828, 0.9895594, -0.0100225, 0.0079734
1: -0.0045499, -0.0039660, -0.0045259, -0.0038655, -0.0006844, 0.0005600
2: 0.0109635, 0.0140580, 0.0104310, 0.0139312, -0.0029676, 0.0036270
3: -0.0077923, -0.0062632, -0.0077089, -0.0060209, -0.0017714, 0.0014457
4: 0.0026499, 0.0037704, 0.0025468, 0.0036348, -0.0009849, 0.0012236
5: 0.0127487, 0.0220452, 0.0120789, 0.0207345, -0.0079858, 0.0099663
6: -0.0026827, -0.0016949, -0.0026423, -0.0015249, -0.0011578, 0.0009473
7: -0.0100787, -0.0075229, -0.0099740, -0.0070831, -0.0029956, 0.0024511
8: -0.0048644, -0.0021174, -0.0048094, -0.0026475, -0.0022170, 0.0026919
9: 0.0022182, 0.0037767, 0.0019500, 0.0037128, -0.0014947, 0.0018267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065166
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065166
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9811702, 0.9896010, -0.0110641, 0.0081635
1: -0.0045644, -0.0039217, -0.0045261, -0.0038551, -0.0007093, 0.0006044
2: 0.0107291, 0.0141350, 0.0103760, 0.0139321, -0.0032031, 0.0037590
3: -0.0078430, -0.0061565, -0.0077095, -0.0059958, -0.0018471, 0.0015530
4: 0.0026045, 0.0038528, 0.0025361, 0.0036358, -0.0010313, 0.0013167
5: 0.0124538, 0.0228415, 0.0120098, 0.0207445, -0.0082907, 0.0108317
6: -0.0027073, -0.0016201, -0.0026426, -0.0015074, -0.0012000, 0.0010225
7: -0.0101423, -0.0073292, -0.0099748, -0.0070377, -0.0031047, 0.0026455
8: -0.0048979, -0.0017954, -0.0048098, -0.0026434, -0.0022545, 0.0030144
9: 0.0021001, 0.0038155, 0.0019223, 0.0037133, -0.0016132, 0.0018932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065995
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065995
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9829116, 0.9894547, -0.0112484, 0.0063318
1: -0.0045692, -0.0039185, -0.0045008, -0.0038916, -0.0006777, 0.0005823
2: 0.0108483, 0.0141605, 0.0105692, 0.0137980, -0.0029497, 0.0035913
3: -0.0078597, -0.0062108, -0.0076213, -0.0060838, -0.0017760, 0.0014105
4: 0.0026275, 0.0038801, 0.0025735, 0.0034922, -0.0008647, 0.0013065
5: 0.0126037, 0.0231048, 0.0122527, 0.0193577, -0.0067539, 0.0108521
6: -0.0027155, -0.0016397, -0.0025997, -0.0015690, -0.0011464, 0.0009601
7: -0.0101634, -0.0074277, -0.0098640, -0.0071972, -0.0029662, 0.0024363
8: -0.0049090, -0.0016889, -0.0047515, -0.0032043, -0.0017047, 0.0030626
9: 0.0021601, 0.0038283, 0.0020196, 0.0036458, -0.0014856, 0.0018087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065586
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065586
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9828985, 0.9894956, -0.0125967, 0.0065393
1: -0.0045882, -0.0038436, -0.0045010, -0.0038814, -0.0007068, 0.0006574
2: 0.0105915, 0.0142612, 0.0105153, 0.0137990, -0.0032074, 0.0037459
3: -0.0079260, -0.0060939, -0.0076220, -0.0060592, -0.0018667, 0.0015280
4: 0.0025779, 0.0039879, 0.0025631, 0.0034933, -0.0009155, 0.0014247
5: 0.0122808, 0.0241461, 0.0121850, 0.0193681, -0.0070873, 0.0119611
6: -0.0027476, -0.0015165, -0.0026001, -0.0015519, -0.0011958, 0.0010836
7: -0.0102466, -0.0072157, -0.0098648, -0.0071527, -0.0030939, 0.0026491
8: -0.0049527, -0.0012678, -0.0047519, -0.0032001, -0.0017526, 0.0034842
9: 0.0020308, 0.0038791, 0.0019924, 0.0036463, -0.0016154, 0.0018866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066527
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066527
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9811828, 0.9895594, -0.0113530, 0.0080606
1: -0.0045692, -0.0039185, -0.0045259, -0.0038655, -0.0007037, 0.0006074
2: 0.0108483, 0.0141605, 0.0104310, 0.0139312, -0.0030829, 0.0037295
3: -0.0078597, -0.0062108, -0.0077089, -0.0060209, -0.0018389, 0.0014981
4: 0.0026275, 0.0038801, 0.0025468, 0.0036348, -0.0010072, 0.0013333
5: 0.0126037, 0.0231048, 0.0120789, 0.0207345, -0.0081308, 0.0110259
6: -0.0027155, -0.0016397, -0.0026423, -0.0015249, -0.0011905, 0.0010026
7: -0.0101634, -0.0074277, -0.0099740, -0.0070831, -0.0030803, 0.0025463
8: -0.0049090, -0.0016889, -0.0048094, -0.0026475, -0.0022615, 0.0031205
9: 0.0021601, 0.0038283, 0.0019500, 0.0037128, -0.0015527, 0.0018783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065518
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065518
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9811702, 0.9896010, -0.0127022, 0.0082676
1: -0.0045882, -0.0038436, -0.0045261, -0.0038551, -0.0007331, 0.0006825
2: 0.0105915, 0.0142612, 0.0103760, 0.0139321, -0.0033406, 0.0038852
3: -0.0079260, -0.0060939, -0.0077095, -0.0059958, -0.0019302, 0.0016156
4: 0.0025779, 0.0039879, 0.0025361, 0.0036358, -0.0010579, 0.0014517
5: 0.0122808, 0.0241461, 0.0120098, 0.0207445, -0.0084637, 0.0121363
6: -0.0027476, -0.0015165, -0.0026426, -0.0015074, -0.0012402, 0.0011261
7: -0.0102466, -0.0072157, -0.0099748, -0.0070377, -0.0032089, 0.0027591
8: -0.0049527, -0.0012678, -0.0048098, -0.0026434, -0.0023093, 0.0035420
9: 0.0020308, 0.0038791, 0.0019223, 0.0037133, -0.0016825, 0.0019568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066458
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066458
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9785167, 0.9893657, -0.0098287, 0.0106394
1: -0.0045499, -0.0039660, -0.0045647, -0.0039137, -0.0006361, 0.0005987
2: 0.0109635, 0.0140580, 0.0106867, 0.0141366, -0.0031731, 0.0033713
3: -0.0077923, -0.0062632, -0.0078440, -0.0061372, -0.0016551, 0.0015808
4: 0.0026499, 0.0037704, 0.0025963, 0.0038545, -0.0012046, 0.0011741
5: 0.0127487, 0.0220452, 0.0124005, 0.0228577, -0.0101090, 0.0096447
6: -0.0026827, -0.0016949, -0.0027078, -0.0016066, -0.0010762, 0.0010129
7: -0.0100787, -0.0075229, -0.0101436, -0.0072943, -0.0027844, 0.0026207
8: -0.0048644, -0.0021174, -0.0048986, -0.0017888, -0.0030756, 0.0027812
9: 0.0022182, 0.0037767, 0.0020788, 0.0038163, -0.0015981, 0.0016979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0065338
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0065338
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9785030, 0.9894069, -0.0108699, 0.0108306
1: -0.0045644, -0.0039217, -0.0045649, -0.0039035, -0.0006609, 0.0006432
2: 0.0107291, 0.0141350, 0.0106324, 0.0141376, -0.0034086, 0.0035026
3: -0.0078430, -0.0061565, -0.0078447, -0.0061126, -0.0017304, 0.0016882
4: 0.0026045, 0.0038528, 0.0025858, 0.0038556, -0.0012512, 0.0012671
5: 0.0124538, 0.0228415, 0.0123323, 0.0228685, -0.0104148, 0.0105092
6: -0.0027073, -0.0016201, -0.0027082, -0.0015892, -0.0011181, 0.0010881
7: -0.0101423, -0.0073292, -0.0101445, -0.0072494, -0.0028929, 0.0028153
8: -0.0048979, -0.0017954, -0.0048990, -0.0017844, -0.0031134, 0.0031037
9: 0.0021001, 0.0038155, 0.0020514, 0.0038168, -0.0017167, 0.0017641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0066240
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0066239
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9768788, 0.9894645, -0.0099275, 0.0122774
1: -0.0045499, -0.0039660, -0.0045885, -0.0038425, -0.0007074, 0.0006226
2: 0.0109635, 0.0140580, 0.0105563, 0.0142628, -0.0032993, 0.0035017
3: -0.0077923, -0.0062632, -0.0079270, -0.0060779, -0.0017144, 0.0016638
4: 0.0026499, 0.0037704, 0.0025710, 0.0039895, -0.0013397, 0.0011994
5: 0.0127487, 0.0220452, 0.0122365, 0.0241622, -0.0114135, 0.0098087
6: -0.0026827, -0.0016949, -0.0027481, -0.0015146, -0.0011681, 0.0010532
7: -0.0100787, -0.0075229, -0.0102479, -0.0071866, -0.0028921, 0.0027250
8: -0.0048644, -0.0021174, -0.0049534, -0.0012613, -0.0036032, 0.0028360
9: 0.0022182, 0.0037767, 0.0020131, 0.0038799, -0.0016617, 0.0017636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0064981
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0064981
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9768654, 0.9895065, -0.0109695, 0.0124683
1: -0.0045644, -0.0039217, -0.0045887, -0.0038417, -0.0007227, 0.0006670
2: 0.0107291, 0.0141350, 0.0105009, 0.0142638, -0.0035347, 0.0036341
3: -0.0078430, -0.0061565, -0.0079277, -0.0060527, -0.0017903, 0.0017712
4: 0.0026045, 0.0038528, 0.0025603, 0.0039906, -0.0013861, 0.0012925
5: 0.0124538, 0.0228415, 0.0121668, 0.0241728, -0.0117190, 0.0106747
6: -0.0027073, -0.0016201, -0.0027485, -0.0015134, -0.0011940, 0.0011284
7: -0.0101423, -0.0073292, -0.0102487, -0.0071408, -0.0030015, 0.0029195
8: -0.0048979, -0.0017954, -0.0049538, -0.0012570, -0.0036409, 0.0031585
9: 0.0021001, 0.0038155, 0.0019852, 0.0038804, -0.0017803, 0.0018303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0065848
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0065848
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9785167, 0.9893657, -0.0111593, 0.0107267
1: -0.0045692, -0.0039185, -0.0045647, -0.0039137, -0.0006555, 0.0006462
2: 0.0108483, 0.0141605, 0.0106867, 0.0141366, -0.0032883, 0.0034738
3: -0.0078597, -0.0062108, -0.0078440, -0.0061372, -0.0017225, 0.0016332
4: 0.0026275, 0.0038801, 0.0025963, 0.0038545, -0.0012270, 0.0012838
5: 0.0126037, 0.0231048, 0.0124005, 0.0228577, -0.0102540, 0.0107043
6: -0.0027155, -0.0016397, -0.0027078, -0.0016066, -0.0011089, 0.0010682
7: -0.0101634, -0.0074277, -0.0101436, -0.0072943, -0.0028691, 0.0027159
8: -0.0049090, -0.0016889, -0.0048986, -0.0017888, -0.0031201, 0.0032097
9: 0.0021601, 0.0038283, 0.0020788, 0.0038163, -0.0016561, 0.0017496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065402
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065402
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9785030, 0.9894069, -0.0125080, 0.0109348
1: -0.0045882, -0.0038436, -0.0045649, -0.0039035, -0.0006847, 0.0007213
2: 0.0105915, 0.0142612, 0.0106324, 0.0141376, -0.0035461, 0.0036288
3: -0.0079260, -0.0060939, -0.0078447, -0.0061126, -0.0018134, 0.0017508
4: 0.0025779, 0.0039879, 0.0025858, 0.0038556, -0.0012778, 0.0014021
5: 0.0122808, 0.0241461, 0.0123323, 0.0228685, -0.0105877, 0.0118138
6: -0.0027476, -0.0015165, -0.0027082, -0.0015892, -0.0011584, 0.0011917
7: -0.0102466, -0.0072157, -0.0101445, -0.0072494, -0.0029971, 0.0029288
8: -0.0049527, -0.0012678, -0.0048990, -0.0017844, -0.0031683, 0.0036313
9: 0.0020308, 0.0038791, 0.0020514, 0.0038168, -0.0017860, 0.0018276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066381
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066381
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9768788, 0.9894645, -0.0112581, 0.0123646
1: -0.0045692, -0.0039185, -0.0045885, -0.0038425, -0.0007268, 0.0006700
2: 0.0108483, 0.0141605, 0.0105563, 0.0142628, -0.0034145, 0.0036042
3: -0.0078597, -0.0062108, -0.0079270, -0.0060779, -0.0017818, 0.0017162
4: 0.0026275, 0.0038801, 0.0025710, 0.0039895, -0.0013620, 0.0013090
5: 0.0126037, 0.0231048, 0.0122365, 0.0241622, -0.0115585, 0.0108683
6: -0.0027155, -0.0016397, -0.0027481, -0.0015146, -0.0012009, 0.0011085
7: -0.0101634, -0.0074277, -0.0102479, -0.0071866, -0.0029768, 0.0028202
8: -0.0049090, -0.0016889, -0.0049534, -0.0012613, -0.0036477, 0.0032645
9: 0.0021601, 0.0038283, 0.0020131, 0.0038799, -0.0017197, 0.0018152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065330
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065330
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9768654, 0.9895065, -0.0126076, 0.0125725
1: -0.0045882, -0.0038436, -0.0045887, -0.0038417, -0.0007465, 0.0007451
2: 0.0105915, 0.0142612, 0.0105009, 0.0142638, -0.0036723, 0.0037603
3: -0.0079260, -0.0060939, -0.0079277, -0.0060527, -0.0018733, 0.0018338
4: 0.0025779, 0.0039879, 0.0025603, 0.0039906, -0.0014128, 0.0014275
5: 0.0122808, 0.0241461, 0.0121668, 0.0241728, -0.0118920, 0.0119793
6: -0.0027476, -0.0015165, -0.0027485, -0.0015134, -0.0012343, 0.0012319
7: -0.0102466, -0.0072157, -0.0102487, -0.0071408, -0.0031058, 0.0030331
8: -0.0049527, -0.0012678, -0.0049538, -0.0012570, -0.0036957, 0.0036861
9: 0.0020308, 0.0038791, 0.0019852, 0.0038804, -0.0018495, 0.0018939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066300
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066300
time: 1.04 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066030
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066030
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066834
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066030, upper bound: 0.0066834
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0065653
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0065653
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0066424
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0066144, upper bound: 0.0066424
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066144
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066144
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0067005
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0067005
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066063
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066063
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066916
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065653, upper bound: 0.0066916
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0065358
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0065358
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0066207
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065544, upper bound: 0.0066207
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0064983
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0064983
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0065788
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065586, upper bound: 0.0065788
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065469
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065469
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066375
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066375
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065396
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0065396
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066296
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065166, upper bound: 0.0066296
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0065544
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0065544
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0066396
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065358, upper bound: 0.0066396
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065166
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065166
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065995
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065469, upper bound: 0.0065995
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065586
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065586
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066527
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066527
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065518
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0065518
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066458
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064983, upper bound: 0.0066458
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0065338
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0065338
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0066240
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065195, upper bound: 0.0066239
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0064981
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0064981
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0065848
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0065280, upper bound: 0.0065848
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065402
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065402
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066381
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066381
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065330
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0065330
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066300
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -0.0064840, upper bound: 0.0066300

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9838164, 0.9892508, -0.0054345, 0.0054345
1: -0.0044877, -0.0039424, -0.0044877, -0.0039424, -0.0005453, 0.0005453
2: 0.0108384, 0.0137283, 0.0108384, 0.0137283, -0.0028898, 0.0028898
3: -0.0075754, -0.0062063, -0.0075754, -0.0062063, -0.0013691, 0.0013691
4: 0.0026256, 0.0034177, 0.0026256, 0.0034177, -0.0007921, 0.0007921
5: 0.0125913, 0.0186372, 0.0125913, 0.0186372, -0.0060458, 0.0060458
6: -0.0025775, -0.0016550, -0.0025775, -0.0016550, -0.0009225, 0.0009225
7: -0.0098064, -0.0074196, -0.0098064, -0.0074196, -0.0023868, 0.0023868
8: -0.0047212, -0.0034660, -0.0047212, -0.0034660, -0.0012552, 0.0012552
9: 0.0021552, 0.0036106, 0.0021552, 0.0036106, -0.0014555, 0.0014555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064032, upper bound: 0.0063966
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063966
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9829309, 0.9894184, -0.0056021, 0.0063199
1: -0.0044877, -0.0039424, -0.0045005, -0.0039006, -0.0005871, 0.0005582
2: 0.0108384, 0.0137283, 0.0106171, 0.0137965, -0.0029581, 0.0031111
3: -0.0075754, -0.0062063, -0.0076203, -0.0061056, -0.0014699, 0.0014140
4: 0.0026256, 0.0034177, 0.0025828, 0.0034907, -0.0008650, 0.0008349
5: 0.0125913, 0.0186372, 0.0123130, 0.0193423, -0.0067510, 0.0063241
6: -0.0025775, -0.0016550, -0.0025993, -0.0015843, -0.0009931, 0.0009443
7: -0.0098064, -0.0074196, -0.0098627, -0.0072368, -0.0025696, 0.0024432
8: -0.0047212, -0.0034660, -0.0047509, -0.0032105, -0.0015107, 0.0012848
9: 0.0021552, 0.0036106, 0.0020437, 0.0036450, -0.0014898, 0.0015669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064032, upper bound: 0.0063966
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063966
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9838164, 0.9892508, -0.0063199, 0.0056021
1: -0.0045005, -0.0039006, -0.0044877, -0.0039424, -0.0005582, 0.0005871
2: 0.0106171, 0.0137965, 0.0108384, 0.0137283, -0.0031111, 0.0029581
3: -0.0076203, -0.0061056, -0.0075754, -0.0062063, -0.0014140, 0.0014699
4: 0.0025828, 0.0034907, 0.0026256, 0.0034177, -0.0008349, 0.0008650
5: 0.0123130, 0.0193423, 0.0125913, 0.0186372, -0.0063241, 0.0067510
6: -0.0025993, -0.0015843, -0.0025775, -0.0016550, -0.0009443, 0.0009931
7: -0.0098627, -0.0072368, -0.0098064, -0.0074196, -0.0024432, 0.0025696
8: -0.0047509, -0.0032105, -0.0047212, -0.0034660, -0.0012848, 0.0015107
9: 0.0020437, 0.0036450, 0.0021552, 0.0036106, -0.0015669, 0.0014898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064018, upper bound: 0.0064731
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9829309, 0.9894184, -0.0064875, 0.0064875
1: -0.0045005, -0.0039006, -0.0045005, -0.0039006, -0.0005999, 0.0005999
2: 0.0106171, 0.0137965, 0.0106171, 0.0137965, -0.0031793, 0.0031793
3: -0.0076203, -0.0061056, -0.0076203, -0.0061056, -0.0015147, 0.0015147
4: 0.0025828, 0.0034907, 0.0025828, 0.0034907, -0.0009079, 0.0009079
5: 0.0123130, 0.0193423, 0.0123130, 0.0193423, -0.0070293, 0.0070293
6: -0.0025993, -0.0015843, -0.0025993, -0.0015843, -0.0010149, 0.0010149
7: -0.0098627, -0.0072368, -0.0098627, -0.0072368, -0.0026259, 0.0026259
8: -0.0047509, -0.0032105, -0.0047509, -0.0032105, -0.0015404, 0.0015404
9: 0.0020437, 0.0036450, 0.0020437, 0.0036450, -0.0016013, 0.0016013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064018, upper bound: 0.0064731
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9822857, 0.9893453, -0.0055289, 0.0069652
1: -0.0044877, -0.0039424, -0.0045099, -0.0039188, -0.0005689, 0.0005676
2: 0.0108384, 0.0137283, 0.0107136, 0.0138462, -0.0030078, 0.0030146
3: -0.0075754, -0.0062063, -0.0076530, -0.0061495, -0.0014259, 0.0014467
4: 0.0026256, 0.0034177, 0.0026015, 0.0035439, -0.0009182, 0.0008162
5: 0.0125913, 0.0186372, 0.0124344, 0.0198562, -0.0072649, 0.0062028
6: -0.0025775, -0.0016550, -0.0026151, -0.0016151, -0.0009624, 0.0009602
7: -0.0098064, -0.0074196, -0.0099038, -0.0073165, -0.0024899, 0.0024842
8: -0.0047212, -0.0034660, -0.0047725, -0.0030027, -0.0017186, 0.0013064
9: 0.0021552, 0.0036106, 0.0020923, 0.0036700, -0.0015149, 0.0015183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064142, upper bound: 0.0063634
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064115, upper bound: 0.0063642
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9811993, 0.9895298, -0.0057134, 0.0080516
1: -0.0044877, -0.0039424, -0.0045257, -0.0038729, -0.0006148, 0.0005833
2: 0.0108384, 0.0137283, 0.0104701, 0.0139299, -0.0030915, 0.0032581
3: -0.0075754, -0.0062063, -0.0077081, -0.0060387, -0.0015368, 0.0015018
4: 0.0026256, 0.0034177, 0.0025544, 0.0036334, -0.0010078, 0.0008633
5: 0.0125913, 0.0186372, 0.0121281, 0.0207213, -0.0081300, 0.0065091
6: -0.0025775, -0.0016550, -0.0026419, -0.0015374, -0.0010401, 0.0009869
7: -0.0098064, -0.0074196, -0.0099729, -0.0071154, -0.0026910, 0.0025534
8: -0.0047212, -0.0034660, -0.0048088, -0.0026528, -0.0020684, 0.0013428
9: 0.0021552, 0.0036106, 0.0019697, 0.0037122, -0.0015570, 0.0016410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064142, upper bound: 0.0063634
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064115, upper bound: 0.0063642
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9822857, 0.9893453, -0.0064144, 0.0071328
1: -0.0045005, -0.0039006, -0.0045099, -0.0039188, -0.0005817, 0.0006093
2: 0.0106171, 0.0137965, 0.0107136, 0.0138462, -0.0032290, 0.0030829
3: -0.0076203, -0.0061056, -0.0076530, -0.0061495, -0.0014708, 0.0015474
4: 0.0025828, 0.0034907, 0.0026015, 0.0035439, -0.0009610, 0.0008892
5: 0.0123130, 0.0193423, 0.0124344, 0.0198562, -0.0075432, 0.0069080
6: -0.0025993, -0.0015843, -0.0026151, -0.0016151, -0.0009841, 0.0010308
7: -0.0098627, -0.0072368, -0.0099038, -0.0073165, -0.0025462, 0.0026670
8: -0.0047509, -0.0032105, -0.0047725, -0.0030027, -0.0017482, 0.0015620
9: 0.0020437, 0.0036450, 0.0020923, 0.0036700, -0.0016263, 0.0015527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064128, upper bound: 0.0064393
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063951, upper bound: 0.0064416
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9811993, 0.9895298, -0.0065989, 0.0082192
1: -0.0045005, -0.0039006, -0.0045257, -0.0038729, -0.0006277, 0.0006251
2: 0.0106171, 0.0137965, 0.0104701, 0.0139299, -0.0033127, 0.0033264
3: -0.0076203, -0.0061056, -0.0077081, -0.0060387, -0.0015817, 0.0016025
4: 0.0025828, 0.0034907, 0.0025544, 0.0036334, -0.0010506, 0.0009363
5: 0.0123130, 0.0193423, 0.0121281, 0.0207213, -0.0084083, 0.0072142
6: -0.0025993, -0.0015843, -0.0026419, -0.0015374, -0.0010619, 0.0010575
7: -0.0098627, -0.0072368, -0.0099729, -0.0071154, -0.0027474, 0.0027361
8: -0.0047509, -0.0032105, -0.0048088, -0.0026528, -0.0020981, 0.0015983
9: 0.0020437, 0.0036450, 0.0019697, 0.0037122, -0.0016685, 0.0016753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064128, upper bound: 0.0064393
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064109, upper bound: 0.0064416
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9838164, 0.9892508, -0.0069652, 0.0055289
1: -0.0045099, -0.0039188, -0.0044877, -0.0039424, -0.0005676, 0.0005689
2: 0.0107136, 0.0138462, 0.0108384, 0.0137283, -0.0030146, 0.0030078
3: -0.0076530, -0.0061495, -0.0075754, -0.0062063, -0.0014467, 0.0014259
4: 0.0026015, 0.0035439, 0.0026256, 0.0034177, -0.0008162, 0.0009182
5: 0.0124344, 0.0198562, 0.0125913, 0.0186372, -0.0062028, 0.0072649
6: -0.0026151, -0.0016151, -0.0025775, -0.0016550, -0.0009602, 0.0009624
7: -0.0099038, -0.0073165, -0.0098064, -0.0074196, -0.0024842, 0.0024899
8: -0.0047725, -0.0030027, -0.0047212, -0.0034660, -0.0013064, 0.0017186
9: 0.0020923, 0.0036700, 0.0021552, 0.0036106, -0.0015183, 0.0015149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064093
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064109
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9829309, 0.9894184, -0.0071328, 0.0064144
1: -0.0045099, -0.0039188, -0.0045005, -0.0039006, -0.0006093, 0.0005817
2: 0.0107136, 0.0138462, 0.0106171, 0.0137965, -0.0030829, 0.0032290
3: -0.0076530, -0.0061495, -0.0076203, -0.0061056, -0.0015474, 0.0014708
4: 0.0026015, 0.0035439, 0.0025828, 0.0034907, -0.0008892, 0.0009610
5: 0.0124344, 0.0198562, 0.0123130, 0.0193423, -0.0069080, 0.0075432
6: -0.0026151, -0.0016151, -0.0025993, -0.0015843, -0.0010308, 0.0009841
7: -0.0099038, -0.0073165, -0.0098627, -0.0072368, -0.0026670, 0.0025462
8: -0.0047725, -0.0030027, -0.0047509, -0.0032105, -0.0015620, 0.0017482
9: 0.0020923, 0.0036700, 0.0020437, 0.0036450, -0.0015527, 0.0016263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064093
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064109
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9838164, 0.9892508, -0.0080516, 0.0057134
1: -0.0045257, -0.0038729, -0.0044877, -0.0039424, -0.0005833, 0.0006148
2: 0.0104701, 0.0139299, 0.0108384, 0.0137283, -0.0032581, 0.0030915
3: -0.0077081, -0.0060387, -0.0075754, -0.0062063, -0.0015018, 0.0015368
4: 0.0025544, 0.0036334, 0.0026256, 0.0034177, -0.0008633, 0.0010078
5: 0.0121281, 0.0207213, 0.0125913, 0.0186372, -0.0065091, 0.0081300
6: -0.0026419, -0.0015374, -0.0025775, -0.0016550, -0.0009869, 0.0010401
7: -0.0099729, -0.0071154, -0.0098064, -0.0074196, -0.0025534, 0.0026910
8: -0.0048088, -0.0026528, -0.0047212, -0.0034660, -0.0013428, 0.0020684
9: 0.0019697, 0.0037122, 0.0021552, 0.0036106, -0.0016410, 0.0015570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064881
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064929
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9829309, 0.9894184, -0.0082192, 0.0065989
1: -0.0045257, -0.0038729, -0.0045005, -0.0039006, -0.0006251, 0.0006277
2: 0.0104701, 0.0139299, 0.0106171, 0.0137965, -0.0033264, 0.0033127
3: -0.0077081, -0.0060387, -0.0076203, -0.0061056, -0.0016025, 0.0015817
4: 0.0025544, 0.0036334, 0.0025828, 0.0034907, -0.0009363, 0.0010506
5: 0.0121281, 0.0207213, 0.0123130, 0.0193423, -0.0072142, 0.0084083
6: -0.0026419, -0.0015374, -0.0025993, -0.0015843, -0.0010575, 0.0010619
7: -0.0099729, -0.0071154, -0.0098627, -0.0072368, -0.0027361, 0.0027474
8: -0.0048088, -0.0026528, -0.0047509, -0.0032105, -0.0015983, 0.0020981
9: 0.0019697, 0.0037122, 0.0020437, 0.0036450, -0.0016753, 0.0016685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064881
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064929
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9822857, 0.9893453, -0.0070596, 0.0070596
1: -0.0045099, -0.0039188, -0.0045099, -0.0039188, -0.0005911, 0.0005911
2: 0.0107136, 0.0138462, 0.0107136, 0.0138462, -0.0031326, 0.0031326
3: -0.0076530, -0.0061495, -0.0076530, -0.0061495, -0.0015035, 0.0015035
4: 0.0026015, 0.0035439, 0.0026015, 0.0035439, -0.0009424, 0.0009424
5: 0.0124344, 0.0198562, 0.0124344, 0.0198562, -0.0074218, 0.0074218
6: -0.0026151, -0.0016151, -0.0026151, -0.0016151, -0.0010000, 0.0010000
7: -0.0099038, -0.0073165, -0.0099038, -0.0073165, -0.0025873, 0.0025873
8: -0.0047725, -0.0030027, -0.0047725, -0.0030027, -0.0017698, 0.0017698
9: 0.0020923, 0.0036700, 0.0020923, 0.0036700, -0.0015777, 0.0015777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064025
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064052
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9811993, 0.9895298, -0.0072441, 0.0081460
1: -0.0045099, -0.0039188, -0.0045257, -0.0038729, -0.0006371, 0.0006069
2: 0.0107136, 0.0138462, 0.0104701, 0.0139299, -0.0032163, 0.0033761
3: -0.0076530, -0.0061495, -0.0077081, -0.0060387, -0.0016143, 0.0015586
4: 0.0026015, 0.0035439, 0.0025544, 0.0036334, -0.0010319, 0.0009895
5: 0.0124344, 0.0198562, 0.0121281, 0.0207213, -0.0082870, 0.0077281
6: -0.0026151, -0.0016151, -0.0026419, -0.0015374, -0.0010777, 0.0010267
7: -0.0099038, -0.0073165, -0.0099729, -0.0071154, -0.0027884, 0.0026564
8: -0.0047725, -0.0030027, -0.0048088, -0.0026528, -0.0021196, 0.0018061
9: 0.0020923, 0.0036700, 0.0019697, 0.0037122, -0.0016199, 0.0017004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064025
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064052
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9822857, 0.9893453, -0.0081460, 0.0072441
1: -0.0045257, -0.0038729, -0.0045099, -0.0039188, -0.0006069, 0.0006371
2: 0.0104701, 0.0139299, 0.0107136, 0.0138462, -0.0033761, 0.0032163
3: -0.0077081, -0.0060387, -0.0076530, -0.0061495, -0.0015586, 0.0016143
4: 0.0025544, 0.0036334, 0.0026015, 0.0035439, -0.0009895, 0.0010319
5: 0.0121281, 0.0207213, 0.0124344, 0.0198562, -0.0077281, 0.0082870
6: -0.0026419, -0.0015374, -0.0026151, -0.0016151, -0.0010267, 0.0010777
7: -0.0099729, -0.0071154, -0.0099038, -0.0073165, -0.0026564, 0.0027884
8: -0.0048088, -0.0026528, -0.0047725, -0.0030027, -0.0018061, 0.0021196
9: 0.0019697, 0.0037122, 0.0020923, 0.0036700, -0.0017004, 0.0016199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064803
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064875
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9811993, 0.9895298, -0.0083305, 0.0083305
1: -0.0045257, -0.0038729, -0.0045257, -0.0038729, -0.0006529, 0.0006529
2: 0.0104701, 0.0139299, 0.0104701, 0.0139299, -0.0034598, 0.0034598
3: -0.0077081, -0.0060387, -0.0077081, -0.0060387, -0.0016694, 0.0016694
4: 0.0025544, 0.0036334, 0.0025544, 0.0036334, -0.0010790, 0.0010790
5: 0.0121281, 0.0207213, 0.0121281, 0.0207213, -0.0085932, 0.0085932
6: -0.0026419, -0.0015374, -0.0026419, -0.0015374, -0.0011045, 0.0011045
7: -0.0099729, -0.0071154, -0.0099729, -0.0071154, -0.0028576, 0.0028576
8: -0.0048088, -0.0026528, -0.0048088, -0.0026528, -0.0021560, 0.0021560
9: 0.0019697, 0.0037122, 0.0019697, 0.0037122, -0.0017425, 0.0017425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064803
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064875
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9795370, 0.9891561, -0.0053397, 0.0097139
1: -0.0044877, -0.0039424, -0.0045499, -0.0039660, -0.0005217, 0.0006075
2: 0.0108384, 0.0137283, 0.0109635, 0.0140580, -0.0032195, 0.0027647
3: -0.0075754, -0.0062063, -0.0077923, -0.0062632, -0.0013122, 0.0015860
4: 0.0026256, 0.0034177, 0.0026499, 0.0037704, -0.0011448, 0.0007678
5: 0.0125913, 0.0186372, 0.0127487, 0.0220452, -0.0094539, 0.0058885
6: -0.0025775, -0.0016550, -0.0026827, -0.0016949, -0.0008826, 0.0010278
7: -0.0098064, -0.0074196, -0.0100787, -0.0075229, -0.0022835, 0.0026591
8: -0.0047212, -0.0034660, -0.0048644, -0.0021174, -0.0026038, 0.0013984
9: 0.0021552, 0.0036106, 0.0022182, 0.0037767, -0.0016215, 0.0013925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063542, upper bound: 0.0063296
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0063296
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9785370, 0.9893336, -0.0055172, 0.0107139
1: -0.0044877, -0.0039424, -0.0045644, -0.0039217, -0.0005659, 0.0006221
2: 0.0108384, 0.0137283, 0.0107291, 0.0141350, -0.0032966, 0.0029992
3: -0.0075754, -0.0062063, -0.0078430, -0.0061565, -0.0014189, 0.0016367
4: 0.0026256, 0.0034177, 0.0026045, 0.0038528, -0.0012272, 0.0008132
5: 0.0125913, 0.0186372, 0.0124538, 0.0228415, -0.0102502, 0.0061834
6: -0.0025775, -0.0016550, -0.0027073, -0.0016201, -0.0009574, 0.0010524
7: -0.0098064, -0.0074196, -0.0101423, -0.0073292, -0.0024772, 0.0027228
8: -0.0047212, -0.0034660, -0.0048979, -0.0017954, -0.0029258, 0.0014319
9: 0.0021552, 0.0036106, 0.0021001, 0.0038155, -0.0016603, 0.0015106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063542, upper bound: 0.0063296
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0063296
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9795370, 0.9891561, -0.0062252, 0.0098815
1: -0.0045005, -0.0039006, -0.0045499, -0.0039660, -0.0005346, 0.0006493
2: 0.0106171, 0.0137965, 0.0109635, 0.0140580, -0.0034408, 0.0028330
3: -0.0076203, -0.0061056, -0.0077923, -0.0062632, -0.0013571, 0.0016867
4: 0.0025828, 0.0034907, 0.0026499, 0.0037704, -0.0011876, 0.0008408
5: 0.0123130, 0.0193423, 0.0127487, 0.0220452, -0.0097322, 0.0065937
6: -0.0025993, -0.0015843, -0.0026827, -0.0016949, -0.0009044, 0.0010984
7: -0.0098627, -0.0072368, -0.0100787, -0.0075229, -0.0023398, 0.0028419
8: -0.0047509, -0.0032105, -0.0048644, -0.0021174, -0.0026334, 0.0016539
9: 0.0020437, 0.0036450, 0.0022182, 0.0037767, -0.0017330, 0.0014268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063541, upper bound: 0.0064110
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0064116
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9785370, 0.9893336, -0.0064027, 0.0108815
1: -0.0045005, -0.0039006, -0.0045644, -0.0039217, -0.0005788, 0.0006638
2: 0.0106171, 0.0137965, 0.0107291, 0.0141350, -0.0035179, 0.0030674
3: -0.0076203, -0.0061056, -0.0078430, -0.0061565, -0.0014638, 0.0017374
4: 0.0025828, 0.0034907, 0.0026045, 0.0038528, -0.0012700, 0.0008862
5: 0.0123130, 0.0193423, 0.0124538, 0.0228415, -0.0105285, 0.0068886
6: -0.0025993, -0.0015843, -0.0027073, -0.0016201, -0.0009792, 0.0011230
7: -0.0098627, -0.0072368, -0.0101423, -0.0073292, -0.0025335, 0.0029055
8: -0.0047509, -0.0032105, -0.0048979, -0.0017954, -0.0029555, 0.0016874
9: 0.0020437, 0.0036450, 0.0021001, 0.0038155, -0.0017718, 0.0015449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063541, upper bound: 0.0064110
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0064116
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9782064, 0.9892434, -0.0054270, 0.0110444
1: -0.0044877, -0.0039424, -0.0045692, -0.0039185, -0.0005691, 0.0006269
2: 0.0108384, 0.0137283, 0.0108483, 0.0141605, -0.0033221, 0.0028800
3: -0.0075754, -0.0062063, -0.0078597, -0.0062108, -0.0013647, 0.0016534
4: 0.0026256, 0.0034177, 0.0026275, 0.0038801, -0.0012544, 0.0007901
5: 0.0125913, 0.0186372, 0.0126037, 0.0231048, -0.0105135, 0.0060334
6: -0.0025775, -0.0016550, -0.0027155, -0.0016397, -0.0009378, 0.0010605
7: -0.0098064, -0.0074196, -0.0101634, -0.0074277, -0.0023787, 0.0027438
8: -0.0047212, -0.0034660, -0.0049090, -0.0016889, -0.0030323, 0.0014429
9: 0.0021552, 0.0036106, 0.0021601, 0.0038283, -0.0016732, 0.0014505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063600, upper bound: 0.0062935
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063508, upper bound: 0.0062935
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9838164, 0.9892508, 0.9768988, 0.9894378, -0.0056214, 0.0123520
1: -0.0044877, -0.0039424, -0.0045882, -0.0038436, -0.0006441, 0.0006459
2: 0.0108384, 0.0137283, 0.0105915, 0.0142612, -0.0034228, 0.0031367
3: -0.0075754, -0.0062063, -0.0079260, -0.0060939, -0.0014815, 0.0017197
4: 0.0026256, 0.0034177, 0.0025779, 0.0039879, -0.0013622, 0.0008398
5: 0.0125913, 0.0186372, 0.0122808, 0.0241461, -0.0115548, 0.0063564
6: -0.0025775, -0.0016550, -0.0027476, -0.0015165, -0.0010610, 0.0010927
7: -0.0098064, -0.0074196, -0.0102466, -0.0072157, -0.0025907, 0.0028270
8: -0.0047212, -0.0034660, -0.0049527, -0.0012678, -0.0034534, 0.0014867
9: 0.0021552, 0.0036106, 0.0020308, 0.0038791, -0.0017239, 0.0015798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063600, upper bound: 0.0062935
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063508, upper bound: 0.0062935
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9782064, 0.9892434, -0.0063125, 0.0112121
1: -0.0045005, -0.0039006, -0.0045692, -0.0039185, -0.0005820, 0.0006686
2: 0.0106171, 0.0137965, 0.0108483, 0.0141605, -0.0035433, 0.0029482
3: -0.0076203, -0.0061056, -0.0078597, -0.0062108, -0.0014095, 0.0017542
4: 0.0025828, 0.0034907, 0.0026275, 0.0038801, -0.0012973, 0.0008631
5: 0.0123130, 0.0193423, 0.0126037, 0.0231048, -0.0107918, 0.0067386
6: -0.0025993, -0.0015843, -0.0027155, -0.0016397, -0.0009596, 0.0011311
7: -0.0098627, -0.0072368, -0.0101634, -0.0074277, -0.0024350, 0.0029266
8: -0.0047509, -0.0032105, -0.0049090, -0.0016889, -0.0030620, 0.0016985
9: 0.0020437, 0.0036450, 0.0021601, 0.0038283, -0.0017846, 0.0014849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0063727
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063507, upper bound: 0.0063727
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9829309, 0.9894184, 0.9768988, 0.9894378, -0.0065069, 0.0125196
1: -0.0045005, -0.0039006, -0.0045882, -0.0038436, -0.0006569, 0.0006876
2: 0.0106171, 0.0137965, 0.0105915, 0.0142612, -0.0036441, 0.0032049
3: -0.0076203, -0.0061056, -0.0079260, -0.0060939, -0.0015264, 0.0018204
4: 0.0025828, 0.0034907, 0.0025779, 0.0039879, -0.0014050, 0.0009128
5: 0.0123130, 0.0193423, 0.0122808, 0.0241461, -0.0118331, 0.0070615
6: -0.0025993, -0.0015843, -0.0027476, -0.0015165, -0.0010828, 0.0011633
7: -0.0098627, -0.0072368, -0.0102466, -0.0072157, -0.0026471, 0.0030098
8: -0.0047509, -0.0032105, -0.0049527, -0.0012678, -0.0034831, 0.0017422
9: 0.0020437, 0.0036450, 0.0020308, 0.0038791, -0.0018353, 0.0016142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0063727
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063507, upper bound: 0.0063727
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9795370, 0.9891561, -0.0068704, 0.0098084
1: -0.0045099, -0.0039188, -0.0045499, -0.0039660, -0.0005440, 0.0006311
2: 0.0107136, 0.0138462, 0.0109635, 0.0140580, -0.0033443, 0.0028827
3: -0.0076530, -0.0061495, -0.0077923, -0.0062632, -0.0013898, 0.0016428
4: 0.0026015, 0.0035439, 0.0026499, 0.0037704, -0.0011689, 0.0008940
5: 0.0124344, 0.0198562, 0.0127487, 0.0220452, -0.0096108, 0.0071075
6: -0.0026151, -0.0016151, -0.0026827, -0.0016949, -0.0009202, 0.0010676
7: -0.0099038, -0.0073165, -0.0100787, -0.0075229, -0.0023809, 0.0027622
8: -0.0047725, -0.0030027, -0.0048644, -0.0021174, -0.0026550, 0.0018618
9: 0.0020923, 0.0036700, 0.0022182, 0.0037767, -0.0016844, 0.0014519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063396
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063396
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9785370, 0.9893336, -0.0070480, 0.0108083
1: -0.0045099, -0.0039188, -0.0045644, -0.0039217, -0.0005882, 0.0006456
2: 0.0107136, 0.0138462, 0.0107291, 0.0141350, -0.0034214, 0.0031171
3: -0.0076530, -0.0061495, -0.0078430, -0.0061565, -0.0014965, 0.0016935
4: 0.0026015, 0.0035439, 0.0026045, 0.0038528, -0.0012513, 0.0009394
5: 0.0124344, 0.0198562, 0.0124538, 0.0228415, -0.0104071, 0.0074024
6: -0.0026151, -0.0016151, -0.0027073, -0.0016201, -0.0009951, 0.0010922
7: -0.0099038, -0.0073165, -0.0101423, -0.0073292, -0.0025746, 0.0028258
8: -0.0047725, -0.0030027, -0.0048979, -0.0017954, -0.0029771, 0.0018952
9: 0.0020923, 0.0036700, 0.0021001, 0.0038155, -0.0017232, 0.0015699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063396
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063396
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9795370, 0.9891561, -0.0079569, 0.0099928
1: -0.0045257, -0.0038729, -0.0045499, -0.0039660, -0.0005597, 0.0006770
2: 0.0104701, 0.0139299, 0.0109635, 0.0140580, -0.0035879, 0.0029664
3: -0.0077081, -0.0060387, -0.0077923, -0.0062632, -0.0014448, 0.0017536
4: 0.0025544, 0.0036334, 0.0026499, 0.0037704, -0.0012160, 0.0009835
5: 0.0121281, 0.0207213, 0.0127487, 0.0220452, -0.0099171, 0.0079727
6: -0.0026419, -0.0015374, -0.0026827, -0.0016949, -0.0009469, 0.0011453
7: -0.0099729, -0.0071154, -0.0100787, -0.0075229, -0.0024500, 0.0029633
8: -0.0048088, -0.0026528, -0.0048644, -0.0021174, -0.0026914, 0.0022116
9: 0.0019697, 0.0037122, 0.0022182, 0.0037767, -0.0018070, 0.0014940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064255
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064270
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9785370, 0.9893336, -0.0081344, 0.0109928
1: -0.0045257, -0.0038729, -0.0045644, -0.0039217, -0.0006040, 0.0006916
2: 0.0104701, 0.0139299, 0.0107291, 0.0141350, -0.0036649, 0.0032008
3: -0.0077081, -0.0060387, -0.0078430, -0.0061565, -0.0015515, 0.0018043
4: 0.0025544, 0.0036334, 0.0026045, 0.0038528, -0.0012985, 0.0010289
5: 0.0121281, 0.0207213, 0.0124538, 0.0228415, -0.0107134, 0.0082676
6: -0.0026419, -0.0015374, -0.0027073, -0.0016201, -0.0010218, 0.0011699
7: -0.0099729, -0.0071154, -0.0101423, -0.0073292, -0.0026437, 0.0030270
8: -0.0048088, -0.0026528, -0.0048979, -0.0017954, -0.0030134, 0.0022451
9: 0.0019697, 0.0037122, 0.0021001, 0.0038155, -0.0018458, 0.0016121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064255
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064270
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9782064, 0.9892434, -0.0069577, 0.0111389
1: -0.0045099, -0.0039188, -0.0045692, -0.0039185, -0.0005914, 0.0006504
2: 0.0107136, 0.0138462, 0.0108483, 0.0141605, -0.0034469, 0.0029979
3: -0.0076530, -0.0061495, -0.0078597, -0.0062108, -0.0014422, 0.0017102
4: 0.0026015, 0.0035439, 0.0026275, 0.0038801, -0.0012786, 0.0009163
5: 0.0124344, 0.0198562, 0.0126037, 0.0231048, -0.0106704, 0.0072525
6: -0.0026151, -0.0016151, -0.0027155, -0.0016397, -0.0009755, 0.0011003
7: -0.0099038, -0.0073165, -0.0101634, -0.0074277, -0.0024761, 0.0028469
8: -0.0047725, -0.0030027, -0.0049090, -0.0016889, -0.0030836, 0.0019063
9: 0.0020923, 0.0036700, 0.0021601, 0.0038283, -0.0017360, 0.0015099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063341
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063341
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9822857, 0.9893453, 0.9768988, 0.9894378, -0.0071521, 0.0124465
1: -0.0045099, -0.0039188, -0.0045882, -0.0038436, -0.0006663, 0.0006694
2: 0.0107136, 0.0138462, 0.0105915, 0.0142612, -0.0035476, 0.0032547
3: -0.0076530, -0.0061495, -0.0079260, -0.0060939, -0.0015591, 0.0017765
4: 0.0026015, 0.0035439, 0.0025779, 0.0039879, -0.0013864, 0.0009660
5: 0.0124344, 0.0198562, 0.0122808, 0.0241461, -0.0117117, 0.0075754
6: -0.0026151, -0.0016151, -0.0027476, -0.0015165, -0.0010986, 0.0011325
7: -0.0099038, -0.0073165, -0.0102466, -0.0072157, -0.0026881, 0.0029301
8: -0.0047725, -0.0030027, -0.0049527, -0.0012678, -0.0035047, 0.0019500
9: 0.0020923, 0.0036700, 0.0020308, 0.0038791, -0.0017867, 0.0016392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063341
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063341
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9782064, 0.9892434, -0.0080441, 0.0113234
1: -0.0045257, -0.0038729, -0.0045692, -0.0039185, -0.0006072, 0.0006964
2: 0.0104701, 0.0139299, 0.0108483, 0.0141605, -0.0036904, 0.0030816
3: -0.0077081, -0.0060387, -0.0078597, -0.0062108, -0.0014973, 0.0018211
4: 0.0025544, 0.0036334, 0.0026275, 0.0038801, -0.0013257, 0.0010058
5: 0.0121281, 0.0207213, 0.0126037, 0.0231048, -0.0109767, 0.0081176
6: -0.0026419, -0.0015374, -0.0027155, -0.0016397, -0.0010022, 0.0011781
7: -0.0099729, -0.0071154, -0.0101634, -0.0074277, -0.0025452, 0.0030480
8: -0.0048088, -0.0026528, -0.0049090, -0.0016889, -0.0031199, 0.0022562
9: 0.0019697, 0.0037122, 0.0021601, 0.0038283, -0.0018587, 0.0015521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064197
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064215
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9811993, 0.9895298, 0.9768988, 0.9894378, -0.0082386, 0.0126309
1: -0.0045257, -0.0038729, -0.0045882, -0.0038436, -0.0006821, 0.0007154
2: 0.0104701, 0.0139299, 0.0105915, 0.0142612, -0.0037911, 0.0033384
3: -0.0077081, -0.0060387, -0.0079260, -0.0060939, -0.0016141, 0.0018873
4: 0.0025544, 0.0036334, 0.0025779, 0.0039879, -0.0014335, 0.0010555
5: 0.0121281, 0.0207213, 0.0122808, 0.0241461, -0.0120180, 0.0084405
6: -0.0026419, -0.0015374, -0.0027476, -0.0015165, -0.0011254, 0.0012102
7: -0.0099729, -0.0071154, -0.0102466, -0.0072157, -0.0027573, 0.0031312
8: -0.0048088, -0.0026528, -0.0049527, -0.0012678, -0.0035410, 0.0022999
9: 0.0019697, 0.0037122, 0.0020308, 0.0038791, -0.0019094, 0.0016814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064197
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064215
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9838164, 0.9892508, -0.0097139, 0.0053397
1: -0.0045499, -0.0039660, -0.0044877, -0.0039424, -0.0006075, 0.0005217
2: 0.0109635, 0.0140580, 0.0108384, 0.0137283, -0.0027647, 0.0032195
3: -0.0077923, -0.0062632, -0.0075754, -0.0062063, -0.0015860, 0.0013122
4: 0.0026499, 0.0037704, 0.0026256, 0.0034177, -0.0007678, 0.0011448
5: 0.0127487, 0.0220452, 0.0125913, 0.0186372, -0.0058885, 0.0094539
6: -0.0026827, -0.0016949, -0.0025775, -0.0016550, -0.0010278, 0.0008826
7: -0.0100787, -0.0075229, -0.0098064, -0.0074196, -0.0026591, 0.0022835
8: -0.0048644, -0.0021174, -0.0047212, -0.0034660, -0.0013984, 0.0026038
9: 0.0022182, 0.0037767, 0.0021552, 0.0036106, -0.0013925, 0.0016215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063336, upper bound: 0.0063463
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063299, upper bound: 0.0063467
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9829309, 0.9894184, -0.0098815, 0.0062252
1: -0.0045499, -0.0039660, -0.0045005, -0.0039006, -0.0006493, 0.0005346
2: 0.0109635, 0.0140580, 0.0106171, 0.0137965, -0.0028330, 0.0034408
3: -0.0077923, -0.0062632, -0.0076203, -0.0061056, -0.0016867, 0.0013571
4: 0.0026499, 0.0037704, 0.0025828, 0.0034907, -0.0008408, 0.0011876
5: 0.0127487, 0.0220452, 0.0123130, 0.0193423, -0.0065937, 0.0097322
6: -0.0026827, -0.0016949, -0.0025993, -0.0015843, -0.0010984, 0.0009044
7: -0.0100787, -0.0075229, -0.0098627, -0.0072368, -0.0028419, 0.0023398
8: -0.0048644, -0.0021174, -0.0047509, -0.0032105, -0.0016539, 0.0026334
9: 0.0022182, 0.0037767, 0.0020437, 0.0036450, -0.0014268, 0.0017330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063336, upper bound: 0.0063463
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063299, upper bound: 0.0063467
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9838164, 0.9892508, -0.0107139, 0.0055172
1: -0.0045644, -0.0039217, -0.0044877, -0.0039424, -0.0006221, 0.0005659
2: 0.0107291, 0.0141350, 0.0108384, 0.0137283, -0.0029992, 0.0032966
3: -0.0078430, -0.0061565, -0.0075754, -0.0062063, -0.0016367, 0.0014189
4: 0.0026045, 0.0038528, 0.0026256, 0.0034177, -0.0008132, 0.0012272
5: 0.0124538, 0.0228415, 0.0125913, 0.0186372, -0.0061834, 0.0102502
6: -0.0027073, -0.0016201, -0.0025775, -0.0016550, -0.0010524, 0.0009574
7: -0.0101423, -0.0073292, -0.0098064, -0.0074196, -0.0027228, 0.0024772
8: -0.0048979, -0.0017954, -0.0047212, -0.0034660, -0.0014319, 0.0029258
9: 0.0021001, 0.0038155, 0.0021552, 0.0036106, -0.0015106, 0.0016603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063334, upper bound: 0.0064300
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063296, upper bound: 0.0064314
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9829309, 0.9894184, -0.0108815, 0.0064027
1: -0.0045644, -0.0039217, -0.0045005, -0.0039006, -0.0006638, 0.0005788
2: 0.0107291, 0.0141350, 0.0106171, 0.0137965, -0.0030674, 0.0035179
3: -0.0078430, -0.0061565, -0.0076203, -0.0061056, -0.0017374, 0.0014638
4: 0.0026045, 0.0038528, 0.0025828, 0.0034907, -0.0008862, 0.0012700
5: 0.0124538, 0.0228415, 0.0123130, 0.0193423, -0.0068886, 0.0105285
6: -0.0027073, -0.0016201, -0.0025993, -0.0015843, -0.0011230, 0.0009792
7: -0.0101423, -0.0073292, -0.0098627, -0.0072368, -0.0029055, 0.0025335
8: -0.0048979, -0.0017954, -0.0047509, -0.0032105, -0.0016874, 0.0029555
9: 0.0021001, 0.0038155, 0.0020437, 0.0036450, -0.0015449, 0.0017718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063334, upper bound: 0.0064300
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063296, upper bound: 0.0064314
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9822857, 0.9893453, -0.0098084, 0.0068704
1: -0.0045499, -0.0039660, -0.0045099, -0.0039188, -0.0006311, 0.0005440
2: 0.0109635, 0.0140580, 0.0107136, 0.0138462, -0.0028827, 0.0033443
3: -0.0077923, -0.0062632, -0.0076530, -0.0061495, -0.0016428, 0.0013898
4: 0.0026499, 0.0037704, 0.0026015, 0.0035439, -0.0008940, 0.0011689
5: 0.0127487, 0.0220452, 0.0124344, 0.0198562, -0.0071075, 0.0096108
6: -0.0026827, -0.0016949, -0.0026151, -0.0016151, -0.0010676, 0.0009202
7: -0.0100787, -0.0075229, -0.0099038, -0.0073165, -0.0027622, 0.0023809
8: -0.0048644, -0.0021174, -0.0047725, -0.0030027, -0.0018618, 0.0026550
9: 0.0022182, 0.0037767, 0.0020923, 0.0036700, -0.0014519, 0.0016844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063460, upper bound: 0.0063104
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063399, upper bound: 0.0063112
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9811993, 0.9895298, -0.0099928, 0.0079569
1: -0.0045499, -0.0039660, -0.0045257, -0.0038729, -0.0006770, 0.0005597
2: 0.0109635, 0.0140580, 0.0104701, 0.0139299, -0.0029664, 0.0035879
3: -0.0077923, -0.0062632, -0.0077081, -0.0060387, -0.0017536, 0.0014448
4: 0.0026499, 0.0037704, 0.0025544, 0.0036334, -0.0009835, 0.0012160
5: 0.0127487, 0.0220452, 0.0121281, 0.0207213, -0.0079727, 0.0099171
6: -0.0026827, -0.0016949, -0.0026419, -0.0015374, -0.0011453, 0.0009469
7: -0.0100787, -0.0075229, -0.0099729, -0.0071154, -0.0029633, 0.0024500
8: -0.0048644, -0.0021174, -0.0048088, -0.0026528, -0.0022116, 0.0026914
9: 0.0022182, 0.0037767, 0.0019697, 0.0037122, -0.0014940, 0.0018070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063460, upper bound: 0.0063104
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063399, upper bound: 0.0063112
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9822857, 0.9893453, -0.0108083, 0.0070480
1: -0.0045644, -0.0039217, -0.0045099, -0.0039188, -0.0006456, 0.0005882
2: 0.0107291, 0.0141350, 0.0107136, 0.0138462, -0.0031171, 0.0034214
3: -0.0078430, -0.0061565, -0.0076530, -0.0061495, -0.0016935, 0.0014965
4: 0.0026045, 0.0038528, 0.0026015, 0.0035439, -0.0009394, 0.0012513
5: 0.0124538, 0.0228415, 0.0124344, 0.0198562, -0.0074024, 0.0104071
6: -0.0027073, -0.0016201, -0.0026151, -0.0016151, -0.0010922, 0.0009951
7: -0.0101423, -0.0073292, -0.0099038, -0.0073165, -0.0028258, 0.0025746
8: -0.0048979, -0.0017954, -0.0047725, -0.0030027, -0.0018952, 0.0029771
9: 0.0021001, 0.0038155, 0.0020923, 0.0036700, -0.0015699, 0.0017232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063458, upper bound: 0.0063940
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063396, upper bound: 0.0063944
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9811993, 0.9895298, -0.0109928, 0.0081344
1: -0.0045644, -0.0039217, -0.0045257, -0.0038729, -0.0006916, 0.0006040
2: 0.0107291, 0.0141350, 0.0104701, 0.0139299, -0.0032008, 0.0036649
3: -0.0078430, -0.0061565, -0.0077081, -0.0060387, -0.0018043, 0.0015515
4: 0.0026045, 0.0038528, 0.0025544, 0.0036334, -0.0010289, 0.0012985
5: 0.0124538, 0.0228415, 0.0121281, 0.0207213, -0.0082676, 0.0107134
6: -0.0027073, -0.0016201, -0.0026419, -0.0015374, -0.0011699, 0.0010218
7: -0.0101423, -0.0073292, -0.0099729, -0.0071154, -0.0030270, 0.0026437
8: -0.0048979, -0.0017954, -0.0048088, -0.0026528, -0.0022451, 0.0030134
9: 0.0021001, 0.0038155, 0.0019697, 0.0037122, -0.0016121, 0.0018458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063458, upper bound: 0.0063940
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063257, upper bound: 0.0063944
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9838164, 0.9892508, -0.0110444, 0.0054270
1: -0.0045692, -0.0039185, -0.0044877, -0.0039424, -0.0006269, 0.0005691
2: 0.0108483, 0.0141605, 0.0108384, 0.0137283, -0.0028800, 0.0033221
3: -0.0078597, -0.0062108, -0.0075754, -0.0062063, -0.0016534, 0.0013647
4: 0.0026275, 0.0038801, 0.0026256, 0.0034177, -0.0007901, 0.0012544
5: 0.0126037, 0.0231048, 0.0125913, 0.0186372, -0.0060334, 0.0105135
6: -0.0027155, -0.0016397, -0.0025775, -0.0016550, -0.0010605, 0.0009378
7: -0.0101634, -0.0074277, -0.0098064, -0.0074196, -0.0027438, 0.0023787
8: -0.0049090, -0.0016889, -0.0047212, -0.0034660, -0.0014429, 0.0030323
9: 0.0021601, 0.0038283, 0.0021552, 0.0036106, -0.0014505, 0.0016732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063474
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063507
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9829309, 0.9894184, -0.0112121, 0.0063125
1: -0.0045692, -0.0039185, -0.0045005, -0.0039006, -0.0006686, 0.0005820
2: 0.0108483, 0.0141605, 0.0106171, 0.0137965, -0.0029482, 0.0035433
3: -0.0078597, -0.0062108, -0.0076203, -0.0061056, -0.0017542, 0.0014095
4: 0.0026275, 0.0038801, 0.0025828, 0.0034907, -0.0008631, 0.0012973
5: 0.0126037, 0.0231048, 0.0123130, 0.0193423, -0.0067386, 0.0107918
6: -0.0027155, -0.0016397, -0.0025993, -0.0015843, -0.0011311, 0.0009596
7: -0.0101634, -0.0074277, -0.0098627, -0.0072368, -0.0029266, 0.0024350
8: -0.0049090, -0.0016889, -0.0047509, -0.0032105, -0.0016985, 0.0030620
9: 0.0021601, 0.0038283, 0.0020437, 0.0036450, -0.0014849, 0.0017846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063474
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063507
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9838164, 0.9892508, -0.0123520, 0.0056214
1: -0.0045882, -0.0038436, -0.0044877, -0.0039424, -0.0006459, 0.0006441
2: 0.0105915, 0.0142612, 0.0108384, 0.0137283, -0.0031367, 0.0034228
3: -0.0079260, -0.0060939, -0.0075754, -0.0062063, -0.0017197, 0.0014815
4: 0.0025779, 0.0039879, 0.0026256, 0.0034177, -0.0008398, 0.0013622
5: 0.0122808, 0.0241461, 0.0125913, 0.0186372, -0.0063564, 0.0115548
6: -0.0027476, -0.0015165, -0.0025775, -0.0016550, -0.0010927, 0.0010610
7: -0.0102466, -0.0072157, -0.0098064, -0.0074196, -0.0028270, 0.0025907
8: -0.0049527, -0.0012678, -0.0047212, -0.0034660, -0.0014867, 0.0034534
9: 0.0020308, 0.0038791, 0.0021552, 0.0036106, -0.0015798, 0.0017239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064399
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064428
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9829309, 0.9894184, -0.0125196, 0.0065069
1: -0.0045882, -0.0038436, -0.0045005, -0.0039006, -0.0006876, 0.0006569
2: 0.0105915, 0.0142612, 0.0106171, 0.0137965, -0.0032049, 0.0036441
3: -0.0079260, -0.0060939, -0.0076203, -0.0061056, -0.0018204, 0.0015264
4: 0.0025779, 0.0039879, 0.0025828, 0.0034907, -0.0009128, 0.0014050
5: 0.0122808, 0.0241461, 0.0123130, 0.0193423, -0.0070615, 0.0118331
6: -0.0027476, -0.0015165, -0.0025993, -0.0015843, -0.0011633, 0.0010828
7: -0.0102466, -0.0072157, -0.0098627, -0.0072368, -0.0030098, 0.0026471
8: -0.0049527, -0.0012678, -0.0047509, -0.0032105, -0.0017422, 0.0034831
9: 0.0020308, 0.0038791, 0.0020437, 0.0036450, -0.0016142, 0.0018353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064399
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064428
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9822857, 0.9893453, -0.0111389, 0.0069577
1: -0.0045692, -0.0039185, -0.0045099, -0.0039188, -0.0006504, 0.0005914
2: 0.0108483, 0.0141605, 0.0107136, 0.0138462, -0.0029979, 0.0034469
3: -0.0078597, -0.0062108, -0.0076530, -0.0061495, -0.0017102, 0.0014422
4: 0.0026275, 0.0038801, 0.0026015, 0.0035439, -0.0009163, 0.0012786
5: 0.0126037, 0.0231048, 0.0124344, 0.0198562, -0.0072525, 0.0106704
6: -0.0027155, -0.0016397, -0.0026151, -0.0016151, -0.0011003, 0.0009755
7: -0.0101634, -0.0074277, -0.0099038, -0.0073165, -0.0028469, 0.0024761
8: -0.0049090, -0.0016889, -0.0047725, -0.0030027, -0.0019063, 0.0030836
9: 0.0021601, 0.0038283, 0.0020923, 0.0036700, -0.0015099, 0.0017360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063403
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063430
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9811993, 0.9895298, -0.0113234, 0.0080441
1: -0.0045692, -0.0039185, -0.0045257, -0.0038729, -0.0006964, 0.0006072
2: 0.0108483, 0.0141605, 0.0104701, 0.0139299, -0.0030816, 0.0036904
3: -0.0078597, -0.0062108, -0.0077081, -0.0060387, -0.0018211, 0.0014973
4: 0.0026275, 0.0038801, 0.0025544, 0.0036334, -0.0010058, 0.0013257
5: 0.0126037, 0.0231048, 0.0121281, 0.0207213, -0.0081176, 0.0109767
6: -0.0027155, -0.0016397, -0.0026419, -0.0015374, -0.0011781, 0.0010022
7: -0.0101634, -0.0074277, -0.0099729, -0.0071154, -0.0030480, 0.0025452
8: -0.0049090, -0.0016889, -0.0048088, -0.0026528, -0.0022562, 0.0031199
9: 0.0021601, 0.0038283, 0.0019697, 0.0037122, -0.0015521, 0.0018587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063403
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063430
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9822857, 0.9893453, -0.0124465, 0.0071521
1: -0.0045882, -0.0038436, -0.0045099, -0.0039188, -0.0006694, 0.0006663
2: 0.0105915, 0.0142612, 0.0107136, 0.0138462, -0.0032547, 0.0035476
3: -0.0079260, -0.0060939, -0.0076530, -0.0061495, -0.0017765, 0.0015591
4: 0.0025779, 0.0039879, 0.0026015, 0.0035439, -0.0009660, 0.0013864
5: 0.0122808, 0.0241461, 0.0124344, 0.0198562, -0.0075754, 0.0117117
6: -0.0027476, -0.0015165, -0.0026151, -0.0016151, -0.0011325, 0.0010986
7: -0.0102466, -0.0072157, -0.0099038, -0.0073165, -0.0029301, 0.0026881
8: -0.0049527, -0.0012678, -0.0047725, -0.0030027, -0.0019500, 0.0035047
9: 0.0020308, 0.0038791, 0.0020923, 0.0036700, -0.0016392, 0.0017867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064336
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064364
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9811993, 0.9895298, -0.0126309, 0.0082386
1: -0.0045882, -0.0038436, -0.0045257, -0.0038729, -0.0007154, 0.0006821
2: 0.0105915, 0.0142612, 0.0104701, 0.0139299, -0.0033384, 0.0037911
3: -0.0079260, -0.0060939, -0.0077081, -0.0060387, -0.0018873, 0.0016141
4: 0.0025779, 0.0039879, 0.0025544, 0.0036334, -0.0010555, 0.0014335
5: 0.0122808, 0.0241461, 0.0121281, 0.0207213, -0.0084405, 0.0120180
6: -0.0027476, -0.0015165, -0.0026419, -0.0015374, -0.0012102, 0.0011254
7: -0.0102466, -0.0072157, -0.0099729, -0.0071154, -0.0031312, 0.0027573
8: -0.0049527, -0.0012678, -0.0048088, -0.0026528, -0.0022999, 0.0035410
9: 0.0020308, 0.0038791, 0.0019697, 0.0037122, -0.0016814, 0.0019094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064336
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064364
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9795370, 0.9891561, -0.0096192, 0.0096192
1: -0.0045499, -0.0039660, -0.0045499, -0.0039660, -0.0005839, 0.0005839
2: 0.0109635, 0.0140580, 0.0109635, 0.0140580, -0.0030944, 0.0030944
3: -0.0077923, -0.0062632, -0.0077923, -0.0062632, -0.0015291, 0.0015291
4: 0.0026499, 0.0037704, 0.0026499, 0.0037704, -0.0011206, 0.0011206
5: 0.0127487, 0.0220452, 0.0127487, 0.0220452, -0.0092965, 0.0092965
6: -0.0026827, -0.0016949, -0.0026827, -0.0016949, -0.0009878, 0.0009878
7: -0.0100787, -0.0075229, -0.0100787, -0.0075229, -0.0025558, 0.0025558
8: -0.0048644, -0.0021174, -0.0048644, -0.0021174, -0.0027470, 0.0027470
9: 0.0022182, 0.0037767, 0.0022182, 0.0037767, -0.0015585, 0.0015585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063200, upper bound: 0.0063224
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0063224
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9785370, 0.9893336, -0.0097967, 0.0106192
1: -0.0045499, -0.0039660, -0.0045644, -0.0039217, -0.0006281, 0.0005984
2: 0.0109635, 0.0140580, 0.0107291, 0.0141350, -0.0031715, 0.0033289
3: -0.0077923, -0.0062632, -0.0078430, -0.0061565, -0.0016358, 0.0015797
4: 0.0026499, 0.0037704, 0.0026045, 0.0038528, -0.0012030, 0.0011659
5: 0.0127487, 0.0220452, 0.0124538, 0.0228415, -0.0100928, 0.0095914
6: -0.0026827, -0.0016949, -0.0027073, -0.0016201, -0.0010627, 0.0010124
7: -0.0100787, -0.0075229, -0.0101423, -0.0073292, -0.0027495, 0.0026194
8: -0.0048644, -0.0021174, -0.0048979, -0.0017954, -0.0030691, 0.0027805
9: 0.0022182, 0.0037767, 0.0021001, 0.0038155, -0.0015973, 0.0016766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063200, upper bound: 0.0063224
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0063224
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9795370, 0.9891561, -0.0106192, 0.0097967
1: -0.0045644, -0.0039217, -0.0045499, -0.0039660, -0.0005984, 0.0006281
2: 0.0107291, 0.0141350, 0.0109635, 0.0140580, -0.0033289, 0.0031715
3: -0.0078430, -0.0061565, -0.0077923, -0.0062632, -0.0015797, 0.0016358
4: 0.0026045, 0.0038528, 0.0026499, 0.0037704, -0.0011659, 0.0012030
5: 0.0124538, 0.0228415, 0.0127487, 0.0220452, -0.0095914, 0.0100928
6: -0.0027073, -0.0016201, -0.0026827, -0.0016949, -0.0010124, 0.0010627
7: -0.0101423, -0.0073292, -0.0100787, -0.0075229, -0.0026194, 0.0027495
8: -0.0048979, -0.0017954, -0.0048644, -0.0021174, -0.0027805, 0.0030691
9: 0.0021001, 0.0038155, 0.0022182, 0.0037767, -0.0016766, 0.0015973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063198, upper bound: 0.0064124
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0064126
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9785370, 0.9893336, -0.0107967, 0.0107967
1: -0.0045644, -0.0039217, -0.0045644, -0.0039217, -0.0006427, 0.0006427
2: 0.0107291, 0.0141350, 0.0107291, 0.0141350, -0.0034059, 0.0034059
3: -0.0078430, -0.0061565, -0.0078430, -0.0061565, -0.0016865, 0.0016865
4: 0.0026045, 0.0038528, 0.0026045, 0.0038528, -0.0012484, 0.0012484
5: 0.0124538, 0.0228415, 0.0124538, 0.0228415, -0.0103877, 0.0103877
6: -0.0027073, -0.0016201, -0.0027073, -0.0016201, -0.0010873, 0.0010873
7: -0.0101423, -0.0073292, -0.0101423, -0.0073292, -0.0028131, 0.0028131
8: -0.0048979, -0.0017954, -0.0048979, -0.0017954, -0.0031025, 0.0031025
9: 0.0021001, 0.0038155, 0.0021001, 0.0038155, -0.0017154, 0.0017154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063198, upper bound: 0.0064124
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0064126
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9782064, 0.9892434, -0.0097064, 0.0109497
1: -0.0045499, -0.0039660, -0.0045692, -0.0039185, -0.0006313, 0.0006033
2: 0.0109635, 0.0140580, 0.0108483, 0.0141605, -0.0031970, 0.0032097
3: -0.0077923, -0.0062632, -0.0078597, -0.0062108, -0.0015815, 0.0015965
4: 0.0026499, 0.0037704, 0.0026275, 0.0038801, -0.0012302, 0.0011429
5: 0.0127487, 0.0220452, 0.0126037, 0.0231048, -0.0103561, 0.0094415
6: -0.0026827, -0.0016949, -0.0027155, -0.0016397, -0.0010431, 0.0010205
7: -0.0100787, -0.0075229, -0.0101634, -0.0074277, -0.0026510, 0.0026405
8: -0.0048644, -0.0021174, -0.0049090, -0.0016889, -0.0031755, 0.0027915
9: 0.0022182, 0.0037767, 0.0021601, 0.0038283, -0.0016101, 0.0016166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063281, upper bound: 0.0062909
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0062910
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9795370, 0.9891561, 0.9768988, 0.9894378, -0.0099009, 0.0122573
1: -0.0045499, -0.0039660, -0.0045882, -0.0038436, -0.0007063, 0.0006223
2: 0.0109635, 0.0140580, 0.0105915, 0.0142612, -0.0032977, 0.0034664
3: -0.0077923, -0.0062632, -0.0079260, -0.0060939, -0.0016984, 0.0016627
4: 0.0026499, 0.0037704, 0.0025779, 0.0039879, -0.0013380, 0.0011925
5: 0.0127487, 0.0220452, 0.0122808, 0.0241461, -0.0113974, 0.0097644
6: -0.0026827, -0.0016949, -0.0027476, -0.0015165, -0.0011662, 0.0010527
7: -0.0100787, -0.0075229, -0.0102466, -0.0072157, -0.0028631, 0.0027237
8: -0.0048644, -0.0021174, -0.0049527, -0.0012678, -0.0035967, 0.0028353
9: 0.0022182, 0.0037767, 0.0020308, 0.0038791, -0.0016609, 0.0017459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063281, upper bound: 0.0062909
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0062910
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9782064, 0.9892434, -0.0107064, 0.0111272
1: -0.0045644, -0.0039217, -0.0045692, -0.0039185, -0.0006459, 0.0006475
2: 0.0107291, 0.0141350, 0.0108483, 0.0141605, -0.0034314, 0.0032867
3: -0.0078430, -0.0061565, -0.0078597, -0.0062108, -0.0016322, 0.0017032
4: 0.0026045, 0.0038528, 0.0026275, 0.0038801, -0.0012756, 0.0012253
5: 0.0124538, 0.0228415, 0.0126037, 0.0231048, -0.0106510, 0.0102378
6: -0.0027073, -0.0016201, -0.0027155, -0.0016397, -0.0010677, 0.0010954
7: -0.0101423, -0.0073292, -0.0101634, -0.0074277, -0.0027146, 0.0028341
8: -0.0048979, -0.0017954, -0.0049090, -0.0016889, -0.0032090, 0.0031136
9: 0.0021001, 0.0038155, 0.0021601, 0.0038283, -0.0017282, 0.0016554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063278, upper bound: 0.0063775
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0063775
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9785370, 0.9893336, 0.9768988, 0.9894378, -0.0109009, 0.0124348
1: -0.0045644, -0.0039217, -0.0045882, -0.0038436, -0.0007208, 0.0006665
2: 0.0107291, 0.0141350, 0.0105915, 0.0142612, -0.0035322, 0.0035435
3: -0.0078430, -0.0061565, -0.0079260, -0.0060939, -0.0017490, 0.0017695
4: 0.0026045, 0.0038528, 0.0025779, 0.0039879, -0.0013834, 0.0012750
5: 0.0124538, 0.0228415, 0.0122808, 0.0241461, -0.0116923, 0.0105607
6: -0.0027073, -0.0016201, -0.0027476, -0.0015165, -0.0011908, 0.0011276
7: -0.0101423, -0.0073292, -0.0102466, -0.0072157, -0.0029267, 0.0029173
8: -0.0048979, -0.0017954, -0.0049527, -0.0012678, -0.0036301, 0.0031573
9: 0.0021001, 0.0038155, 0.0020308, 0.0038791, -0.0017790, 0.0017847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063278, upper bound: 0.0063775
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0063774
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9795370, 0.9891561, -0.0109497, 0.0097064
1: -0.0045692, -0.0039185, -0.0045499, -0.0039660, -0.0006033, 0.0006313
2: 0.0108483, 0.0141605, 0.0109635, 0.0140580, -0.0032097, 0.0031970
3: -0.0078597, -0.0062108, -0.0077923, -0.0062632, -0.0015965, 0.0015815
4: 0.0026275, 0.0038801, 0.0026499, 0.0037704, -0.0011429, 0.0012302
5: 0.0126037, 0.0231048, 0.0127487, 0.0220452, -0.0094415, 0.0103561
6: -0.0027155, -0.0016397, -0.0026827, -0.0016949, -0.0010205, 0.0010431
7: -0.0101634, -0.0074277, -0.0100787, -0.0075229, -0.0026405, 0.0026510
8: -0.0049090, -0.0016889, -0.0048644, -0.0021174, -0.0027915, 0.0031755
9: 0.0021601, 0.0038283, 0.0022182, 0.0037767, -0.0016166, 0.0016101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063280
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063288
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9785370, 0.9893336, -0.0111272, 0.0107064
1: -0.0045692, -0.0039185, -0.0045644, -0.0039217, -0.0006475, 0.0006459
2: 0.0108483, 0.0141605, 0.0107291, 0.0141350, -0.0032867, 0.0034314
3: -0.0078597, -0.0062108, -0.0078430, -0.0061565, -0.0017032, 0.0016322
4: 0.0026275, 0.0038801, 0.0026045, 0.0038528, -0.0012253, 0.0012756
5: 0.0126037, 0.0231048, 0.0124538, 0.0228415, -0.0102378, 0.0106510
6: -0.0027155, -0.0016397, -0.0027073, -0.0016201, -0.0010954, 0.0010677
7: -0.0101634, -0.0074277, -0.0101423, -0.0073292, -0.0028341, 0.0027146
8: -0.0049090, -0.0016889, -0.0048979, -0.0017954, -0.0031136, 0.0032090
9: 0.0021601, 0.0038283, 0.0021001, 0.0038155, -0.0016554, 0.0017282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063280
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063288
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9795370, 0.9891561, -0.0122573, 0.0099009
1: -0.0045882, -0.0038436, -0.0045499, -0.0039660, -0.0006223, 0.0007063
2: 0.0105915, 0.0142612, 0.0109635, 0.0140580, -0.0034664, 0.0032977
3: -0.0079260, -0.0060939, -0.0077923, -0.0062632, -0.0016627, 0.0016984
4: 0.0025779, 0.0039879, 0.0026499, 0.0037704, -0.0011925, 0.0013380
5: 0.0122808, 0.0241461, 0.0127487, 0.0220452, -0.0097644, 0.0113974
6: -0.0027476, -0.0015165, -0.0026827, -0.0016949, -0.0010527, 0.0011662
7: -0.0102466, -0.0072157, -0.0100787, -0.0075229, -0.0027237, 0.0028631
8: -0.0049527, -0.0012678, -0.0048644, -0.0021174, -0.0028353, 0.0035967
9: 0.0020308, 0.0038791, 0.0022182, 0.0037767, -0.0017459, 0.0016609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064253
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064259
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9785370, 0.9893336, -0.0124348, 0.0109009
1: -0.0045882, -0.0038436, -0.0045644, -0.0039217, -0.0006665, 0.0007208
2: 0.0105915, 0.0142612, 0.0107291, 0.0141350, -0.0035435, 0.0035322
3: -0.0079260, -0.0060939, -0.0078430, -0.0061565, -0.0017695, 0.0017490
4: 0.0025779, 0.0039879, 0.0026045, 0.0038528, -0.0012750, 0.0013834
5: 0.0122808, 0.0241461, 0.0124538, 0.0228415, -0.0105607, 0.0116923
6: -0.0027476, -0.0015165, -0.0027073, -0.0016201, -0.0011276, 0.0011908
7: -0.0102466, -0.0072157, -0.0101423, -0.0073292, -0.0029173, 0.0029267
8: -0.0049527, -0.0012678, -0.0048979, -0.0017954, -0.0031573, 0.0036301
9: 0.0020308, 0.0038791, 0.0021001, 0.0038155, -0.0017847, 0.0017790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064253
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064259
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9782064, 0.9892434, -0.0110370, 0.0110370
1: -0.0045692, -0.0039185, -0.0045692, -0.0039185, -0.0006507, 0.0006507
2: 0.0108483, 0.0141605, 0.0108483, 0.0141605, -0.0033122, 0.0033122
3: -0.0078597, -0.0062108, -0.0078597, -0.0062108, -0.0016490, 0.0016490
4: 0.0026275, 0.0038801, 0.0026275, 0.0038801, -0.0012525, 0.0012525
5: 0.0126037, 0.0231048, 0.0126037, 0.0231048, -0.0105011, 0.0105011
6: -0.0027155, -0.0016397, -0.0027155, -0.0016397, -0.0010758, 0.0010758
7: -0.0101634, -0.0074277, -0.0101634, -0.0074277, -0.0027357, 0.0027357
8: -0.0049090, -0.0016889, -0.0049090, -0.0016889, -0.0032201, 0.0032201
9: 0.0021601, 0.0038283, 0.0021601, 0.0038283, -0.0016682, 0.0016682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063225
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063244
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9782064, 0.9892434, 0.9768988, 0.9894378, -0.0112314, 0.0123445
1: -0.0045692, -0.0039185, -0.0045882, -0.0038436, -0.0007256, 0.0006697
2: 0.0108483, 0.0141605, 0.0105915, 0.0142612, -0.0034129, 0.0035689
3: -0.0078597, -0.0062108, -0.0079260, -0.0060939, -0.0017658, 0.0017152
4: 0.0026275, 0.0038801, 0.0025779, 0.0039879, -0.0013603, 0.0013022
5: 0.0126037, 0.0231048, 0.0122808, 0.0241461, -0.0115424, 0.0108240
6: -0.0027155, -0.0016397, -0.0027476, -0.0015165, -0.0011990, 0.0011080
7: -0.0101634, -0.0074277, -0.0102466, -0.0072157, -0.0029477, 0.0028189
8: -0.0049090, -0.0016889, -0.0049527, -0.0012678, -0.0036412, 0.0032638
9: 0.0021601, 0.0038283, 0.0020308, 0.0038791, -0.0017189, 0.0017975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063225
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063244
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9782064, 0.9892434, -0.0123445, 0.0112314
1: -0.0045882, -0.0038436, -0.0045692, -0.0039185, -0.0006697, 0.0007256
2: 0.0105915, 0.0142612, 0.0108483, 0.0141605, -0.0035689, 0.0034129
3: -0.0079260, -0.0060939, -0.0078597, -0.0062108, -0.0017152, 0.0017658
4: 0.0025779, 0.0039879, 0.0026275, 0.0038801, -0.0013022, 0.0013603
5: 0.0122808, 0.0241461, 0.0126037, 0.0231048, -0.0108240, 0.0115424
6: -0.0027476, -0.0015165, -0.0027155, -0.0016397, -0.0011080, 0.0011990
7: -0.0102466, -0.0072157, -0.0101634, -0.0074277, -0.0028189, 0.0029477
8: -0.0049527, -0.0012678, -0.0049090, -0.0016889, -0.0032638, 0.0036412
9: 0.0020308, 0.0038791, 0.0021601, 0.0038283, -0.0017975, 0.0017189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064191
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064204
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9768988, 0.9894378, 0.9768988, 0.9894378, -0.0125390, 0.0125390
1: -0.0045882, -0.0038436, -0.0045882, -0.0038436, -0.0007446, 0.0007446
2: 0.0105915, 0.0142612, 0.0105915, 0.0142612, -0.0036697, 0.0036697
3: -0.0079260, -0.0060939, -0.0079260, -0.0060939, -0.0018321, 0.0018321
4: 0.0025779, 0.0039879, 0.0025779, 0.0039879, -0.0014100, 0.0014100
5: 0.0122808, 0.0241461, 0.0122808, 0.0241461, -0.0118653, 0.0118653
6: -0.0027476, -0.0015165, -0.0027476, -0.0015165, -0.0012311, 0.0012311
7: -0.0102466, -0.0072157, -0.0102466, -0.0072157, -0.0030309, 0.0030309
8: -0.0049527, -0.0012678, -0.0049527, -0.0012678, -0.0036849, 0.0036849
9: 0.0020308, 0.0038791, 0.0020308, 0.0038791, -0.0018482, 0.0018482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064191
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064204
time: 1.26 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064032, upper bound: 0.0063966
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063966
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064032, upper bound: 0.0063966
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063966
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064018, upper bound: 0.0064731
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064018, upper bound: 0.0064731
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064142, upper bound: 0.0063634
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064115, upper bound: 0.0063642
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064142, upper bound: 0.0063634
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064115, upper bound: 0.0063642
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064128, upper bound: 0.0064393
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063951, upper bound: 0.0064416
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064128, upper bound: 0.0064393
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0064109, upper bound: 0.0064416
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064093
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064109
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064093
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064109
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064881
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064929
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064881
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064929
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064025
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064052
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063599, upper bound: 0.0064025
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063652, upper bound: 0.0064052
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064803
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064875
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0064803
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063642, upper bound: 0.0064875
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063542, upper bound: 0.0063296
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0063296
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063542, upper bound: 0.0063296
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0063296
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063541, upper bound: 0.0064110
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0064116
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063541, upper bound: 0.0064110
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063467, upper bound: 0.0064116
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063600, upper bound: 0.0062935
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063508, upper bound: 0.0062935
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063600, upper bound: 0.0062935
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063508, upper bound: 0.0062935
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0063727
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063507, upper bound: 0.0063727
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063598, upper bound: 0.0063727
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063507, upper bound: 0.0063727
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063396
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063396
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063396
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063396
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064255
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064270
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064255
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064270
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063341
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063341
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0063341
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063341
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064197
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064215
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064197
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064215
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063336, upper bound: 0.0063463
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063299, upper bound: 0.0063467
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063336, upper bound: 0.0063463
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063299, upper bound: 0.0063467
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063334, upper bound: 0.0064300
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063296, upper bound: 0.0064314
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063334, upper bound: 0.0064300
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063296, upper bound: 0.0064314
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063460, upper bound: 0.0063104
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063399, upper bound: 0.0063112
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063460, upper bound: 0.0063104
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063399, upper bound: 0.0063112
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063458, upper bound: 0.0063940
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063396, upper bound: 0.0063944
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063458, upper bound: 0.0063940
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063257, upper bound: 0.0063944
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063474
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063507
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063474
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063507
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064399
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064428
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064399
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064428
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063403
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063430
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063403
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063430
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064336
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064364
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064336
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064364
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063200, upper bound: 0.0063224
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0063224
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063200, upper bound: 0.0063224
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0063224
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063198, upper bound: 0.0064124
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0064126
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063198, upper bound: 0.0064124
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0064126
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063281, upper bound: 0.0062909
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0062910
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063281, upper bound: 0.0062909
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0062910
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063278, upper bound: 0.0063775
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0063775
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063278, upper bound: 0.0063775
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0063774
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063280
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063288
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063280
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063288
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064253
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064259
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064253
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064259
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063225
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063244
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063225
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063244
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064191
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064204
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064191
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.68
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064204

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9839675, 0.9892504, -0.0041358, 0.0052889
1: -0.0044688, -0.0039410, -0.0044855, -0.0039425, -0.0005263, 0.0005445
2: 0.0108312, 0.0136282, 0.0108389, 0.0137166, -0.0028854, 0.0027893
3: -0.0075097, -0.0062030, -0.0075678, -0.0062065, -0.0013031, 0.0013648
4: 0.0026242, 0.0033107, 0.0026257, 0.0034052, -0.0007810, 0.0006850
5: 0.0125822, 0.0176034, 0.0125920, 0.0185168, -0.0059346, 0.0050114
6: -0.0025456, -0.0016527, -0.0025738, -0.0016551, -0.0008904, 0.0009211
7: -0.0097238, -0.0074136, -0.0097968, -0.0074200, -0.0023038, 0.0023832
8: -0.0046778, -0.0034629, -0.0047162, -0.0034662, -0.0012116, 0.0012533
9: 0.0021515, 0.0035603, 0.0021554, 0.0036048, -0.0014533, 0.0014048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063933, upper bound: 0.0063933
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063933, upper bound: 0.0063971
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9839411, 0.9892506, -0.0041269, 0.0053069
1: -0.0044687, -0.0039431, -0.0044858, -0.0039424, -0.0005262, 0.0005428
2: 0.0108421, 0.0136275, 0.0108388, 0.0137186, -0.0028765, 0.0027888
3: -0.0075092, -0.0062080, -0.0075691, -0.0062065, -0.0013027, 0.0013611
4: 0.0026264, 0.0033099, 0.0026257, 0.0034074, -0.0007810, 0.0006842
5: 0.0125960, 0.0175961, 0.0125918, 0.0185378, -0.0059418, 0.0050044
6: -0.0025453, -0.0016562, -0.0025744, -0.0016551, -0.0008902, 0.0009182
7: -0.0097232, -0.0074226, -0.0097985, -0.0074199, -0.0023034, 0.0023758
8: -0.0046775, -0.0034676, -0.0047171, -0.0034662, -0.0012113, 0.0012494
9: 0.0021571, 0.0035599, 0.0021553, 0.0036058, -0.0014488, 0.0014046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063933
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063971
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9830698, 0.9894180, -0.0043034, 0.0061865
1: -0.0044688, -0.0039410, -0.0044985, -0.0039007, -0.0005681, 0.0005575
2: 0.0108312, 0.0136282, 0.0106177, 0.0137858, -0.0029546, 0.0030106
3: -0.0075097, -0.0062030, -0.0076133, -0.0061058, -0.0014038, 0.0014103
4: 0.0026242, 0.0033107, 0.0025829, 0.0034792, -0.0008550, 0.0007278
5: 0.0125822, 0.0176034, 0.0123137, 0.0192317, -0.0066495, 0.0052897
6: -0.0025456, -0.0016527, -0.0025959, -0.0015845, -0.0009611, 0.0009432
7: -0.0097238, -0.0074136, -0.0098539, -0.0072372, -0.0024866, 0.0024403
8: -0.0046778, -0.0034629, -0.0047462, -0.0032552, -0.0014226, 0.0012833
9: 0.0021515, 0.0035603, 0.0020440, 0.0036396, -0.0014881, 0.0015163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064708, upper bound: 0.0063920
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064708, upper bound: 0.0063966
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9830661, 0.9894182, -0.0042945, 0.0061818
1: -0.0044687, -0.0039431, -0.0044986, -0.0039007, -0.0005680, 0.0005555
2: 0.0108421, 0.0136275, 0.0106175, 0.0137861, -0.0029439, 0.0030100
3: -0.0075092, -0.0062080, -0.0076135, -0.0061057, -0.0014035, 0.0014055
4: 0.0026264, 0.0033099, 0.0025829, 0.0034795, -0.0008532, 0.0007271
5: 0.0125960, 0.0175961, 0.0123135, 0.0192346, -0.0066386, 0.0052827
6: -0.0025453, -0.0016562, -0.0025959, -0.0015845, -0.0009609, 0.0009398
7: -0.0097232, -0.0074226, -0.0098541, -0.0072371, -0.0024861, 0.0024315
8: -0.0046775, -0.0034676, -0.0047463, -0.0032541, -0.0014234, 0.0012787
9: 0.0021571, 0.0035599, 0.0020439, 0.0036398, -0.0014827, 0.0015160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064731, upper bound: 0.0063920
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064731, upper bound: 0.0063966
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9839675, 0.9892504, -0.0050640, 0.0054606
1: -0.0044823, -0.0038982, -0.0044855, -0.0039425, -0.0005398, 0.0005873
2: 0.0106044, 0.0136997, 0.0108389, 0.0137166, -0.0031122, 0.0028608
3: -0.0075567, -0.0060998, -0.0075678, -0.0062065, -0.0013502, 0.0014680
4: 0.0025803, 0.0033872, 0.0026257, 0.0034052, -0.0008249, 0.0007614
5: 0.0122970, 0.0183425, 0.0125920, 0.0185168, -0.0062198, 0.0057505
6: -0.0025684, -0.0015803, -0.0025738, -0.0016551, -0.0009132, 0.0009935
7: -0.0097828, -0.0072263, -0.0097968, -0.0074200, -0.0023629, 0.0025705
8: -0.0047088, -0.0033644, -0.0047162, -0.0034662, -0.0012426, 0.0013518
9: 0.0020373, 0.0035963, 0.0021554, 0.0036048, -0.0015675, 0.0014409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064708
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064731
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9839411, 0.9892506, -0.0049318, 0.0054745
1: -0.0044804, -0.0039013, -0.0044858, -0.0039424, -0.0005379, 0.0005845
2: 0.0106209, 0.0136895, 0.0108388, 0.0137186, -0.0030977, 0.0028508
3: -0.0075500, -0.0061073, -0.0075691, -0.0062065, -0.0013435, 0.0014618
4: 0.0025836, 0.0033763, 0.0026257, 0.0034074, -0.0008239, 0.0007506
5: 0.0123178, 0.0182370, 0.0125918, 0.0185378, -0.0062200, 0.0056453
6: -0.0025651, -0.0015856, -0.0025744, -0.0016551, -0.0009100, 0.0009889
7: -0.0097744, -0.0072399, -0.0097985, -0.0074199, -0.0023546, 0.0025585
8: -0.0047044, -0.0033716, -0.0047171, -0.0034662, -0.0012383, 0.0013455
9: 0.0020456, 0.0035911, 0.0021553, 0.0036058, -0.0015602, 0.0014358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064708
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9830698, 0.9894180, -0.0052316, 0.0063583
1: -0.0044823, -0.0038982, -0.0044985, -0.0039007, -0.0005816, 0.0006003
2: 0.0106044, 0.0136997, 0.0106177, 0.0137858, -0.0031814, 0.0030821
3: -0.0075567, -0.0060998, -0.0076133, -0.0061058, -0.0014509, 0.0015135
4: 0.0025803, 0.0033872, 0.0025829, 0.0034792, -0.0008989, 0.0008043
5: 0.0122970, 0.0183425, 0.0123137, 0.0192317, -0.0069347, 0.0060288
6: -0.0025684, -0.0015803, -0.0025959, -0.0015845, -0.0009839, 0.0010156
7: -0.0097828, -0.0072263, -0.0098539, -0.0072372, -0.0025456, 0.0026276
8: -0.0047088, -0.0033644, -0.0047462, -0.0032552, -0.0014536, 0.0013818
9: 0.0020373, 0.0035963, 0.0020440, 0.0036396, -0.0016023, 0.0015523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064708
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064731
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9830661, 0.9894182, -0.0050994, 0.0063494
1: -0.0044804, -0.0039013, -0.0044986, -0.0039007, -0.0005797, 0.0005973
2: 0.0106209, 0.0136895, 0.0106175, 0.0137861, -0.0031651, 0.0030720
3: -0.0075500, -0.0061073, -0.0076135, -0.0061057, -0.0014442, 0.0015061
4: 0.0025836, 0.0033763, 0.0025829, 0.0034795, -0.0008960, 0.0007934
5: 0.0123178, 0.0182370, 0.0123135, 0.0192346, -0.0069168, 0.0059236
6: -0.0025651, -0.0015856, -0.0025959, -0.0015845, -0.0009807, 0.0010104
7: -0.0097744, -0.0072399, -0.0098541, -0.0072371, -0.0025373, 0.0026142
8: -0.0047044, -0.0033716, -0.0047463, -0.0032541, -0.0014504, 0.0013748
9: 0.0020456, 0.0035911, 0.0020439, 0.0036398, -0.0015941, 0.0015472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064708
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9824310, 0.9893449, -0.0042303, 0.0068254
1: -0.0044688, -0.0039410, -0.0045078, -0.0039189, -0.0005499, 0.0005668
2: 0.0108312, 0.0136282, 0.0107142, 0.0138350, -0.0030038, 0.0029141
3: -0.0075097, -0.0062030, -0.0076456, -0.0061497, -0.0013599, 0.0014426
4: 0.0026242, 0.0033107, 0.0026016, 0.0035319, -0.0009076, 0.0007091
5: 0.0125822, 0.0176034, 0.0124351, 0.0197404, -0.0071582, 0.0051683
6: -0.0025456, -0.0016527, -0.0026116, -0.0016153, -0.0009302, 0.0009589
7: -0.0097238, -0.0074136, -0.0098945, -0.0073169, -0.0024068, 0.0024810
8: -0.0046778, -0.0034629, -0.0047676, -0.0030495, -0.0016283, 0.0013047
9: 0.0021515, 0.0035603, 0.0020926, 0.0036644, -0.0015129, 0.0014677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064048, upper bound: 0.0063544
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064048, upper bound: 0.0063646
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9824136, 0.9893451, -0.0042214, 0.0068343
1: -0.0044687, -0.0039431, -0.0045081, -0.0039189, -0.0005498, 0.0005650
2: 0.0108421, 0.0136275, 0.0107140, 0.0138363, -0.0029942, 0.0029136
3: -0.0075092, -0.0062080, -0.0076465, -0.0061497, -0.0013595, 0.0014385
4: 0.0026264, 0.0033099, 0.0026016, 0.0035333, -0.0009070, 0.0007084
5: 0.0125960, 0.0175961, 0.0124348, 0.0197543, -0.0071583, 0.0051613
6: -0.0025453, -0.0016562, -0.0026120, -0.0016153, -0.0009301, 0.0009558
7: -0.0097232, -0.0074226, -0.0098957, -0.0073168, -0.0024064, 0.0024730
8: -0.0046775, -0.0034676, -0.0047682, -0.0030439, -0.0016336, 0.0013005
9: 0.0021571, 0.0035599, 0.0020925, 0.0036651, -0.0015080, 0.0014674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064098, upper bound: 0.0063544
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064098, upper bound: 0.0063652
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9813352, 0.9895294, -0.0044148, 0.0079212
1: -0.0044688, -0.0039410, -0.0045237, -0.0038730, -0.0005958, 0.0005827
2: 0.0108312, 0.0136282, 0.0104707, 0.0139194, -0.0030882, 0.0031576
3: -0.0075097, -0.0062030, -0.0077012, -0.0060389, -0.0014708, 0.0014982
4: 0.0026242, 0.0033107, 0.0025545, 0.0036222, -0.0009980, 0.0007562
5: 0.0125822, 0.0176034, 0.0121288, 0.0206131, -0.0080308, 0.0054746
6: -0.0025456, -0.0016527, -0.0026385, -0.0015376, -0.0010080, 0.0009858
7: -0.0097238, -0.0074136, -0.0099643, -0.0071158, -0.0026080, 0.0025507
8: -0.0046778, -0.0034629, -0.0048043, -0.0026966, -0.0019812, 0.0013414
9: 0.0021515, 0.0035603, 0.0019700, 0.0037069, -0.0015554, 0.0015903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064860, upper bound: 0.0063536
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064860, upper bound: 0.0063634
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9813367, 0.9895295, -0.0044058, 0.0079113
1: -0.0044687, -0.0039431, -0.0045237, -0.0038729, -0.0005957, 0.0005807
2: 0.0108421, 0.0136275, 0.0104705, 0.0139193, -0.0030772, 0.0031571
3: -0.0075092, -0.0062080, -0.0077011, -0.0060388, -0.0014704, 0.0014931
4: 0.0026264, 0.0033099, 0.0025544, 0.0036221, -0.0009957, 0.0007555
5: 0.0125960, 0.0175961, 0.0121286, 0.0206119, -0.0080159, 0.0054676
6: -0.0025453, -0.0016562, -0.0026385, -0.0015375, -0.0010078, 0.0009823
7: -0.0097232, -0.0074226, -0.0099642, -0.0071157, -0.0026075, 0.0025415
8: -0.0046775, -0.0034676, -0.0048042, -0.0026971, -0.0019804, 0.0013366
9: 0.0021571, 0.0035599, 0.0019699, 0.0037069, -0.0015498, 0.0015901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064881, upper bound: 0.0063536
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064881, upper bound: 0.0063642
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9824310, 0.9893449, -0.0051585, 0.0069971
1: -0.0044823, -0.0038982, -0.0045078, -0.0039189, -0.0005634, 0.0006096
2: 0.0106044, 0.0136997, 0.0107142, 0.0138350, -0.0032306, 0.0029856
3: -0.0075567, -0.0060998, -0.0076456, -0.0061497, -0.0014069, 0.0015459
4: 0.0025803, 0.0033872, 0.0026016, 0.0035319, -0.0009515, 0.0007856
5: 0.0122970, 0.0183425, 0.0124351, 0.0197404, -0.0074434, 0.0059074
6: -0.0025684, -0.0015803, -0.0026116, -0.0016153, -0.0009531, 0.0010313
7: -0.0097828, -0.0072263, -0.0098945, -0.0073169, -0.0024659, 0.0026683
8: -0.0047088, -0.0033644, -0.0047676, -0.0030495, -0.0016594, 0.0014032
9: 0.0020373, 0.0035963, 0.0020926, 0.0036644, -0.0016271, 0.0015037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064292
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064393
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9824136, 0.9893451, -0.0050263, 0.0070019
1: -0.0044804, -0.0039013, -0.0045081, -0.0039189, -0.0005615, 0.0006067
2: 0.0106209, 0.0136895, 0.0107140, 0.0138363, -0.0032154, 0.0029756
3: -0.0075500, -0.0061073, -0.0076465, -0.0061497, -0.0014003, 0.0015392
4: 0.0025836, 0.0033763, 0.0026016, 0.0035333, -0.0009498, 0.0007747
5: 0.0123178, 0.0182370, 0.0124348, 0.0197543, -0.0074365, 0.0058022
6: -0.0025651, -0.0015856, -0.0026120, -0.0016153, -0.0009499, 0.0010264
7: -0.0097744, -0.0072399, -0.0098957, -0.0073168, -0.0024576, 0.0026557
8: -0.0047044, -0.0033716, -0.0047682, -0.0030439, -0.0016605, 0.0013966
9: 0.0020456, 0.0035911, 0.0020925, 0.0036651, -0.0016194, 0.0014986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064093, upper bound: 0.0064292
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064093, upper bound: 0.0064416
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9813352, 0.9895294, -0.0053430, 0.0080929
1: -0.0044823, -0.0038982, -0.0045237, -0.0038730, -0.0006093, 0.0006255
2: 0.0106044, 0.0136997, 0.0104707, 0.0139194, -0.0033150, 0.0032291
3: -0.0075567, -0.0060998, -0.0077012, -0.0060389, -0.0015178, 0.0016014
4: 0.0025803, 0.0033872, 0.0025545, 0.0036222, -0.0010418, 0.0008327
5: 0.0122970, 0.0183425, 0.0121288, 0.0206131, -0.0083161, 0.0062137
6: -0.0025684, -0.0015803, -0.0026385, -0.0015376, -0.0010308, 0.0010582
7: -0.0097828, -0.0072263, -0.0099643, -0.0071158, -0.0026670, 0.0027380
8: -0.0047088, -0.0033644, -0.0048043, -0.0026966, -0.0020123, 0.0014399
9: 0.0020373, 0.0035963, 0.0019700, 0.0037069, -0.0016696, 0.0016263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064292
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064393
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9813367, 0.9895295, -0.0052107, 0.0080789
1: -0.0044804, -0.0039013, -0.0045237, -0.0038729, -0.0006074, 0.0006224
2: 0.0106209, 0.0136895, 0.0104705, 0.0139193, -0.0032984, 0.0032191
3: -0.0075500, -0.0061073, -0.0077011, -0.0060388, -0.0015111, 0.0015938
4: 0.0025836, 0.0033763, 0.0025544, 0.0036221, -0.0010385, 0.0008218
5: 0.0123178, 0.0182370, 0.0121286, 0.0206119, -0.0082941, 0.0061085
6: -0.0025651, -0.0015856, -0.0026385, -0.0015375, -0.0010276, 0.0010529
7: -0.0097744, -0.0072399, -0.0099642, -0.0071157, -0.0026587, 0.0027242
8: -0.0047044, -0.0033716, -0.0048042, -0.0026971, -0.0020074, 0.0014326
9: 0.0020456, 0.0035911, 0.0019699, 0.0037069, -0.0016612, 0.0016213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063951, upper bound: 0.0064292
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064093, upper bound: 0.0064416
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9839675, 0.9892504, -0.0057768, 0.0053898
1: -0.0044926, -0.0039158, -0.0044855, -0.0039425, -0.0005502, 0.0005696
2: 0.0106979, 0.0137547, 0.0108389, 0.0137166, -0.0030187, 0.0029157
3: -0.0075928, -0.0061423, -0.0075678, -0.0062065, -0.0013863, 0.0014254
4: 0.0025984, 0.0034459, 0.0026257, 0.0034052, -0.0008068, 0.0008202
5: 0.0124146, 0.0189102, 0.0125920, 0.0185168, -0.0061023, 0.0063182
6: -0.0025859, -0.0016101, -0.0025738, -0.0016551, -0.0009308, 0.0009637
7: -0.0098282, -0.0073035, -0.0097968, -0.0074200, -0.0024082, 0.0024933
8: -0.0047327, -0.0033853, -0.0047162, -0.0034662, -0.0012665, 0.0013309
9: 0.0020844, 0.0036239, 0.0021554, 0.0036048, -0.0015204, 0.0014685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0064048
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0064098
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9836155, 0.9893424, 0.9839411, 0.9892506, -0.0056351, 0.0054013
1: -0.0044906, -0.0039195, -0.0044858, -0.0039424, -0.0005482, 0.0005663
2: 0.0107175, 0.0137437, 0.0108388, 0.0137186, -0.0030012, 0.0029050
3: -0.0075856, -0.0061513, -0.0075691, -0.0062065, -0.0013792, 0.0014179
4: 0.0026022, 0.0034343, 0.0026257, 0.0034074, -0.0008052, 0.0008085
5: 0.0124392, 0.0187972, 0.0125918, 0.0185378, -0.0060986, 0.0062054
6: -0.0025824, -0.0016164, -0.0025744, -0.0016551, -0.0009273, 0.0009580
7: -0.0098192, -0.0073197, -0.0097985, -0.0074199, -0.0023993, 0.0024788
8: -0.0047280, -0.0034135, -0.0047171, -0.0034662, -0.0012618, 0.0013036
9: 0.0020943, 0.0036184, 0.0021553, 0.0036058, -0.0015115, 0.0014631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0064048
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0064115
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9830698, 0.9894180, -0.0059444, 0.0062875
1: -0.0044926, -0.0039158, -0.0044985, -0.0039007, -0.0005919, 0.0005827
2: 0.0106979, 0.0137547, 0.0106177, 0.0137858, -0.0030879, 0.0031370
3: -0.0075928, -0.0061423, -0.0076133, -0.0061058, -0.0014870, 0.0014709
4: 0.0025984, 0.0034459, 0.0025829, 0.0034792, -0.0008808, 0.0008630
5: 0.0124146, 0.0189102, 0.0123137, 0.0192317, -0.0068171, 0.0065965
6: -0.0025859, -0.0016101, -0.0025959, -0.0015845, -0.0010014, 0.0009857
7: -0.0098282, -0.0073035, -0.0098539, -0.0072372, -0.0025910, 0.0025504
8: -0.0047327, -0.0033853, -0.0047462, -0.0032552, -0.0014775, 0.0013610
9: 0.0020844, 0.0036239, 0.0020440, 0.0036396, -0.0015552, 0.0015800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0064035
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0064093
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9836155, 0.9893424, 0.9830661, 0.9894182, -0.0058028, 0.0062762
1: -0.0044906, -0.0039195, -0.0044986, -0.0039007, -0.0005899, 0.0005790
2: 0.0107175, 0.0137437, 0.0106175, 0.0137861, -0.0030686, 0.0031262
3: -0.0075856, -0.0061513, -0.0076135, -0.0061057, -0.0014799, 0.0014622
4: 0.0026022, 0.0034343, 0.0025829, 0.0034795, -0.0008773, 0.0008514
5: 0.0124392, 0.0187972, 0.0123135, 0.0192346, -0.0067954, 0.0064837
6: -0.0025824, -0.0016164, -0.0025959, -0.0015845, -0.0009980, 0.0009796
7: -0.0098192, -0.0073197, -0.0098541, -0.0072371, -0.0025821, 0.0025344
8: -0.0047280, -0.0034135, -0.0047463, -0.0032541, -0.0014739, 0.0013328
9: 0.0020943, 0.0036184, 0.0020439, 0.0036398, -0.0015455, 0.0015745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0064035
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0064109
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9823954, 0.9895465, 0.9839675, 0.9892504, -0.0068551, 0.0055790
1: -0.0045083, -0.0038687, -0.0044855, -0.0039425, -0.0005659, 0.0006168
2: 0.0104480, 0.0138377, 0.0108389, 0.0137166, -0.0032686, 0.0029988
3: -0.0076474, -0.0060286, -0.0075678, -0.0062065, -0.0014409, 0.0015392
4: 0.0025501, 0.0035348, 0.0026257, 0.0034052, -0.0008551, 0.0009091
5: 0.0121003, 0.0197688, 0.0125920, 0.0185168, -0.0064165, 0.0071768
6: -0.0026124, -0.0015304, -0.0025738, -0.0016551, -0.0009573, 0.0010434
7: -0.0098968, -0.0070971, -0.0097968, -0.0074200, -0.0024768, 0.0026997
8: -0.0047688, -0.0030380, -0.0047162, -0.0034662, -0.0013025, 0.0016782
9: 0.0019586, 0.0036658, 0.0021554, 0.0036048, -0.0016462, 0.0015104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064860
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064881
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9826351, 0.9895267, 0.9839411, 0.9892506, -0.0066155, 0.0055857
1: -0.0045048, -0.0038736, -0.0044858, -0.0039424, -0.0005624, 0.0006123
2: 0.0104740, 0.0138193, 0.0108388, 0.0137186, -0.0032446, 0.0029805
3: -0.0076353, -0.0060404, -0.0075691, -0.0062065, -0.0014288, 0.0015287
4: 0.0025551, 0.0035150, 0.0026257, 0.0034074, -0.0008523, 0.0008893
5: 0.0121330, 0.0195778, 0.0125918, 0.0185378, -0.0064048, 0.0069861
6: -0.0026065, -0.0015387, -0.0025744, -0.0016551, -0.0009515, 0.0010358
7: -0.0098816, -0.0071186, -0.0097985, -0.0074199, -0.0024617, 0.0026799
8: -0.0047608, -0.0031152, -0.0047171, -0.0034662, -0.0012946, 0.0016018
9: 0.0019716, 0.0036565, 0.0021553, 0.0036058, -0.0016342, 0.0015011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064866
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064929
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9823954, 0.9895465, 0.9830698, 0.9894180, -0.0070226, 0.0064766
1: -0.0045083, -0.0038687, -0.0044985, -0.0039007, -0.0006076, 0.0006298
2: 0.0104480, 0.0138377, 0.0106177, 0.0137858, -0.0033378, 0.0032201
3: -0.0076474, -0.0060286, -0.0076133, -0.0061058, -0.0015416, 0.0015847
4: 0.0025501, 0.0035348, 0.0025829, 0.0034792, -0.0009291, 0.0009519
5: 0.0121003, 0.0197688, 0.0123137, 0.0192317, -0.0071314, 0.0074551
6: -0.0026124, -0.0015304, -0.0025959, -0.0015845, -0.0010279, 0.0010655
7: -0.0098968, -0.0070971, -0.0098539, -0.0072372, -0.0026596, 0.0027568
8: -0.0047688, -0.0030380, -0.0047462, -0.0032552, -0.0015136, 0.0017082
9: 0.0019586, 0.0036658, 0.0020440, 0.0036396, -0.0016811, 0.0016218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064860
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064881
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9826351, 0.9895267, 0.9830661, 0.9894182, -0.0067831, 0.0064606
1: -0.0045048, -0.0038736, -0.0044986, -0.0039007, -0.0006042, 0.0006250
2: 0.0104740, 0.0138193, 0.0106175, 0.0137861, -0.0033120, 0.0032018
3: -0.0076353, -0.0060404, -0.0076135, -0.0061057, -0.0015295, 0.0015730
4: 0.0025551, 0.0035150, 0.0025829, 0.0034795, -0.0009244, 0.0009322
5: 0.0121330, 0.0195778, 0.0123135, 0.0192346, -0.0071016, 0.0072644
6: -0.0026065, -0.0015387, -0.0025959, -0.0015845, -0.0010221, 0.0010573
7: -0.0098816, -0.0071186, -0.0098541, -0.0072371, -0.0026445, 0.0027355
8: -0.0047608, -0.0031152, -0.0047463, -0.0032541, -0.0015067, 0.0016311
9: 0.0019716, 0.0036565, 0.0020439, 0.0036398, -0.0016681, 0.0016126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064866
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064929
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9824310, 0.9893449, -0.0058713, 0.0069263
1: -0.0044926, -0.0039158, -0.0045078, -0.0039189, -0.0005737, 0.0005920
2: 0.0106979, 0.0137547, 0.0107142, 0.0138350, -0.0031371, 0.0030405
3: -0.0075928, -0.0061423, -0.0076456, -0.0061497, -0.0014431, 0.0015033
4: 0.0025984, 0.0034459, 0.0026016, 0.0035319, -0.0009334, 0.0008443
5: 0.0124146, 0.0189102, 0.0124351, 0.0197404, -0.0073259, 0.0064751
6: -0.0025859, -0.0016101, -0.0026116, -0.0016153, -0.0009706, 0.0010014
7: -0.0098282, -0.0073035, -0.0098945, -0.0073169, -0.0025113, 0.0025911
8: -0.0047327, -0.0033853, -0.0047676, -0.0030495, -0.0016832, 0.0013823
9: 0.0020844, 0.0036239, 0.0020926, 0.0036644, -0.0015800, 0.0015314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0063962
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0064030
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9836155, 0.9893424, 0.9824136, 0.9893451, -0.0057296, 0.0069287
1: -0.0044906, -0.0039195, -0.0045081, -0.0039189, -0.0005717, 0.0005885
2: 0.0107175, 0.0137437, 0.0107140, 0.0138363, -0.0031188, 0.0030297
3: -0.0075856, -0.0061513, -0.0076465, -0.0061497, -0.0014359, 0.0014953
4: 0.0026022, 0.0034343, 0.0026016, 0.0035333, -0.0009311, 0.0008327
5: 0.0124392, 0.0187972, 0.0124348, 0.0197543, -0.0073150, 0.0063624
6: -0.0025824, -0.0016164, -0.0026120, -0.0016153, -0.0009672, 0.0009956
7: -0.0098192, -0.0073197, -0.0098957, -0.0073168, -0.0025024, 0.0025760
8: -0.0047280, -0.0034135, -0.0047682, -0.0030439, -0.0016841, 0.0013547
9: 0.0020943, 0.0036184, 0.0020925, 0.0036651, -0.0015708, 0.0015259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0063962
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0064059
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9813352, 0.9895294, -0.0060558, 0.0080221
1: -0.0044926, -0.0039158, -0.0045237, -0.0038730, -0.0006197, 0.0006079
2: 0.0106979, 0.0137547, 0.0104707, 0.0139194, -0.0032215, 0.0032840
3: -0.0075928, -0.0061423, -0.0077012, -0.0060389, -0.0015539, 0.0015588
4: 0.0025984, 0.0034459, 0.0025545, 0.0036222, -0.0010238, 0.0008915
5: 0.0124146, 0.0189102, 0.0121288, 0.0206131, -0.0081985, 0.0067814
6: -0.0025859, -0.0016101, -0.0026385, -0.0015376, -0.0010483, 0.0010284
7: -0.0098282, -0.0073035, -0.0099643, -0.0071158, -0.0027124, 0.0026608
8: -0.0047327, -0.0033853, -0.0048043, -0.0026966, -0.0020361, 0.0014190
9: 0.0020844, 0.0036239, 0.0019700, 0.0037069, -0.0016225, 0.0016540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0063948
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0064025
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9836155, 0.9893424, 0.9813367, 0.9895295, -0.0059140, 0.0080057
1: -0.0044906, -0.0039195, -0.0045237, -0.0038729, -0.0006177, 0.0006042
2: 0.0107175, 0.0137437, 0.0104705, 0.0139193, -0.0032018, 0.0032733
3: -0.0075856, -0.0061513, -0.0077011, -0.0060388, -0.0015468, 0.0015498
4: 0.0026022, 0.0034343, 0.0025544, 0.0036221, -0.0010198, 0.0008798
5: 0.0124392, 0.0187972, 0.0121286, 0.0206119, -0.0081727, 0.0066686
6: -0.0025824, -0.0016164, -0.0026385, -0.0015375, -0.0010449, 0.0010221
7: -0.0098192, -0.0073197, -0.0099642, -0.0071157, -0.0027035, 0.0026445
8: -0.0047280, -0.0034135, -0.0048042, -0.0026971, -0.0020309, 0.0013907
9: 0.0020943, 0.0036184, 0.0019699, 0.0037069, -0.0016126, 0.0016486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0063948
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0064052
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9823954, 0.9895465, 0.9824310, 0.9893449, -0.0069495, 0.0071155
1: -0.0045083, -0.0038687, -0.0045078, -0.0039189, -0.0005894, 0.0006391
2: 0.0104480, 0.0138377, 0.0107142, 0.0138350, -0.0033870, 0.0031236
3: -0.0076474, -0.0060286, -0.0076456, -0.0061497, -0.0014977, 0.0016170
4: 0.0025501, 0.0035348, 0.0026016, 0.0035319, -0.0009818, 0.0009332
5: 0.0121003, 0.0197688, 0.0124351, 0.0197404, -0.0076401, 0.0073337
6: -0.0026124, -0.0015304, -0.0026116, -0.0016153, -0.0009971, 0.0010812
7: -0.0098968, -0.0070971, -0.0098945, -0.0073169, -0.0025799, 0.0027974
8: -0.0047688, -0.0030380, -0.0047676, -0.0030495, -0.0017193, 0.0017296
9: 0.0019586, 0.0036658, 0.0020926, 0.0036644, -0.0017059, 0.0015732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064760
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064803
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9826351, 0.9895267, 0.9824136, 0.9893451, -0.0067099, 0.0071131
1: -0.0045048, -0.0038736, -0.0045081, -0.0039189, -0.0005860, 0.0006345
2: 0.0104740, 0.0138193, 0.0107140, 0.0138363, -0.0033623, 0.0031053
3: -0.0076353, -0.0060404, -0.0076465, -0.0061497, -0.0014856, 0.0016061
4: 0.0025551, 0.0035150, 0.0026016, 0.0035333, -0.0009782, 0.0009135
5: 0.0121330, 0.0195778, 0.0124348, 0.0197543, -0.0076213, 0.0071430
6: -0.0026065, -0.0015387, -0.0026120, -0.0016153, -0.0009913, 0.0010733
7: -0.0098816, -0.0071186, -0.0098957, -0.0073168, -0.0025648, 0.0027771
8: -0.0047608, -0.0031152, -0.0047682, -0.0030439, -0.0017169, 0.0016529
9: 0.0019716, 0.0036565, 0.0020925, 0.0036651, -0.0016934, 0.0015640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064766
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064875
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9823954, 0.9895465, 0.9813352, 0.9895294, -0.0071340, 0.0082113
1: -0.0045083, -0.0038687, -0.0045237, -0.0038730, -0.0006354, 0.0006550
2: 0.0104480, 0.0138377, 0.0104707, 0.0139194, -0.0034714, 0.0033671
3: -0.0076474, -0.0060286, -0.0077012, -0.0060389, -0.0016085, 0.0016726
4: 0.0025501, 0.0035348, 0.0025545, 0.0036222, -0.0010721, 0.0009803
5: 0.0121003, 0.0197688, 0.0121288, 0.0206131, -0.0085128, 0.0076400
6: -0.0026124, -0.0015304, -0.0026385, -0.0015376, -0.0010749, 0.0011082
7: -0.0098968, -0.0070971, -0.0099643, -0.0071158, -0.0027810, 0.0028672
8: -0.0047688, -0.0030380, -0.0048043, -0.0026966, -0.0020722, 0.0017662
9: 0.0019586, 0.0036658, 0.0019700, 0.0037069, -0.0017484, 0.0016958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064760
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064803
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9826351, 0.9895267, 0.9813367, 0.9895295, -0.0068944, 0.0081901
1: -0.0045048, -0.0038736, -0.0045237, -0.0038729, -0.0006319, 0.0006501
2: 0.0104740, 0.0138193, 0.0104705, 0.0139193, -0.0034453, 0.0033488
3: -0.0076353, -0.0060404, -0.0077011, -0.0060388, -0.0015965, 0.0016607
4: 0.0025551, 0.0035150, 0.0025544, 0.0036221, -0.0010670, 0.0009606
5: 0.0121330, 0.0195778, 0.0121286, 0.0206119, -0.0084789, 0.0074493
6: -0.0026065, -0.0015387, -0.0026385, -0.0015375, -0.0010690, 0.0010998
7: -0.0098816, -0.0071186, -0.0099642, -0.0071157, -0.0027659, 0.0028456
8: -0.0047608, -0.0031152, -0.0048042, -0.0026971, -0.0020637, 0.0016890
9: 0.0019716, 0.0036565, 0.0019699, 0.0037069, -0.0017352, 0.0016866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064766
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064875
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9796757, 0.9891557, -0.0040411, 0.0095807
1: -0.0044688, -0.0039410, -0.0045479, -0.0039661, -0.0005027, 0.0006069
2: 0.0108312, 0.0136282, 0.0109641, 0.0140473, -0.0032161, 0.0026642
3: -0.0075097, -0.0062030, -0.0077853, -0.0062635, -0.0012462, 0.0015823
4: 0.0026242, 0.0033107, 0.0026500, 0.0037590, -0.0011347, 0.0006607
5: 0.0125822, 0.0176034, 0.0127494, 0.0219347, -0.0093525, 0.0048540
6: -0.0025456, -0.0016527, -0.0026793, -0.0016951, -0.0008505, 0.0010267
7: -0.0097238, -0.0074136, -0.0100699, -0.0075233, -0.0022004, 0.0026563
8: -0.0046778, -0.0034629, -0.0048598, -0.0021621, -0.0025157, 0.0013969
9: 0.0021515, 0.0035603, 0.0022185, 0.0037713, -0.0016198, 0.0013418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063432, upper bound: 0.0063257
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063432, upper bound: 0.0063299
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9796717, 0.9891558, -0.0040321, 0.0095763
1: -0.0044687, -0.0039431, -0.0045479, -0.0039660, -0.0005026, 0.0006049
2: 0.0108421, 0.0136275, 0.0109639, 0.0140476, -0.0032054, 0.0026637
3: -0.0075092, -0.0062080, -0.0077855, -0.0062634, -0.0012458, 0.0015775
4: 0.0026264, 0.0033099, 0.0026499, 0.0037593, -0.0011329, 0.0006600
5: 0.0125960, 0.0175961, 0.0127491, 0.0219379, -0.0093419, 0.0048470
6: -0.0025453, -0.0016562, -0.0026794, -0.0016950, -0.0008503, 0.0010232
7: -0.0097232, -0.0074226, -0.0100701, -0.0075232, -0.0022000, 0.0026475
8: -0.0046775, -0.0034676, -0.0048599, -0.0021608, -0.0025167, 0.0013923
9: 0.0021571, 0.0035599, 0.0022184, 0.0037715, -0.0016144, 0.0013416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0063257
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0063299
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9786741, 0.9893333, -0.0042187, 0.0105823
1: -0.0044688, -0.0039410, -0.0045624, -0.0039218, -0.0005470, 0.0006214
2: 0.0108312, 0.0136282, 0.0107296, 0.0141244, -0.0032932, 0.0028987
3: -0.0075097, -0.0062030, -0.0078360, -0.0061568, -0.0013529, 0.0016330
4: 0.0026242, 0.0033107, 0.0026046, 0.0038415, -0.0012173, 0.0007061
5: 0.0125822, 0.0176034, 0.0124544, 0.0227323, -0.0101501, 0.0051489
6: -0.0025456, -0.0016527, -0.0027040, -0.0016202, -0.0009253, 0.0010513
7: -0.0097238, -0.0074136, -0.0101336, -0.0073297, -0.0023941, 0.0027200
8: -0.0046778, -0.0034629, -0.0048933, -0.0018395, -0.0028383, 0.0014304
9: 0.0021515, 0.0035603, 0.0021004, 0.0038102, -0.0016587, 0.0014599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064274, upper bound: 0.0063247
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064274, upper bound: 0.0063296
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9786724, 0.9893335, -0.0042098, 0.0105756
1: -0.0044687, -0.0039431, -0.0045624, -0.0039218, -0.0005469, 0.0006194
2: 0.0108421, 0.0136275, 0.0107294, 0.0141246, -0.0032824, 0.0028981
3: -0.0075092, -0.0062080, -0.0078361, -0.0061567, -0.0013525, 0.0016281
4: 0.0026264, 0.0033099, 0.0026045, 0.0038417, -0.0012153, 0.0007054
5: 0.0125960, 0.0175961, 0.0124542, 0.0227336, -0.0101376, 0.0051419
6: -0.0025453, -0.0016562, -0.0027040, -0.0016202, -0.0009252, 0.0010478
7: -0.0097232, -0.0074226, -0.0101337, -0.0073295, -0.0023937, 0.0027111
8: -0.0046775, -0.0034676, -0.0048934, -0.0018390, -0.0028385, 0.0014257
9: 0.0021571, 0.0035599, 0.0021003, 0.0038102, -0.0016532, 0.0014597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064300, upper bound: 0.0063247
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064300, upper bound: 0.0063296
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9796757, 0.9891557, -0.0049692, 0.0097524
1: -0.0044823, -0.0038982, -0.0045479, -0.0039661, -0.0005162, 0.0006497
2: 0.0106044, 0.0136997, 0.0109641, 0.0140473, -0.0034429, 0.0027357
3: -0.0075567, -0.0060998, -0.0077853, -0.0062635, -0.0012932, 0.0016855
4: 0.0025803, 0.0033872, 0.0026500, 0.0037590, -0.0011786, 0.0007372
5: 0.0122970, 0.0183425, 0.0127494, 0.0219347, -0.0096377, 0.0055931
6: -0.0025684, -0.0015803, -0.0026793, -0.0016951, -0.0008733, 0.0010991
7: -0.0097828, -0.0072263, -0.0100699, -0.0075233, -0.0022595, 0.0028436
8: -0.0047088, -0.0033644, -0.0048598, -0.0021621, -0.0025467, 0.0014954
9: 0.0020373, 0.0035963, 0.0022185, 0.0037713, -0.0017340, 0.0013778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064082
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064110
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9796717, 0.9891558, -0.0048370, 0.0097439
1: -0.0044804, -0.0039013, -0.0045479, -0.0039660, -0.0005143, 0.0006466
2: 0.0106209, 0.0136895, 0.0109639, 0.0140476, -0.0034266, 0.0027257
3: -0.0075500, -0.0061073, -0.0077855, -0.0062634, -0.0012866, 0.0016782
4: 0.0025836, 0.0033763, 0.0026499, 0.0037593, -0.0011758, 0.0007263
5: 0.0123178, 0.0182370, 0.0127491, 0.0219379, -0.0096201, 0.0054879
6: -0.0025651, -0.0015856, -0.0026794, -0.0016950, -0.0008701, 0.0010939
7: -0.0097744, -0.0072399, -0.0100701, -0.0075232, -0.0022512, 0.0028302
8: -0.0047044, -0.0033716, -0.0048599, -0.0021608, -0.0025436, 0.0014884
9: 0.0020456, 0.0035911, 0.0022184, 0.0037715, -0.0017258, 0.0013728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064082
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064116
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9786741, 0.9893333, -0.0051469, 0.0107540
1: -0.0044823, -0.0038982, -0.0045624, -0.0039218, -0.0005605, 0.0006642
2: 0.0106044, 0.0136997, 0.0107296, 0.0141244, -0.0035200, 0.0029702
3: -0.0075567, -0.0060998, -0.0078360, -0.0061568, -0.0013999, 0.0017362
4: 0.0025803, 0.0033872, 0.0026046, 0.0038415, -0.0012612, 0.0007826
5: 0.0122970, 0.0183425, 0.0124544, 0.0227323, -0.0104354, 0.0058880
6: -0.0025684, -0.0015803, -0.0027040, -0.0016202, -0.0009482, 0.0011237
7: -0.0097828, -0.0072263, -0.0101336, -0.0073297, -0.0024532, 0.0029073
8: -0.0047088, -0.0033644, -0.0048933, -0.0018395, -0.0028693, 0.0015289
9: 0.0020373, 0.0035963, 0.0021004, 0.0038102, -0.0017729, 0.0014959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064082
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064110
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9786724, 0.9893335, -0.0050147, 0.0107432
1: -0.0044804, -0.0039013, -0.0045624, -0.0039218, -0.0005586, 0.0006611
2: 0.0106209, 0.0136895, 0.0107294, 0.0141246, -0.0035036, 0.0029601
3: -0.0075500, -0.0061073, -0.0078361, -0.0061567, -0.0013933, 0.0017288
4: 0.0025836, 0.0033763, 0.0026045, 0.0038417, -0.0012581, 0.0007717
5: 0.0123178, 0.0182370, 0.0124542, 0.0227336, -0.0104158, 0.0057828
6: -0.0025651, -0.0015856, -0.0027040, -0.0016202, -0.0009450, 0.0011185
7: -0.0097744, -0.0072399, -0.0101337, -0.0073295, -0.0024449, 0.0028938
8: -0.0047044, -0.0033716, -0.0048934, -0.0018390, -0.0028654, 0.0015218
9: 0.0020456, 0.0035911, 0.0021003, 0.0038102, -0.0017646, 0.0014909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064082
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064116
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9783440, 0.9892430, -0.0041284, 0.0109124
1: -0.0044688, -0.0039410, -0.0045672, -0.0039264, -0.0005424, 0.0006262
2: 0.0108312, 0.0136282, 0.0108488, 0.0141499, -0.0033187, 0.0027794
3: -0.0075097, -0.0062030, -0.0078527, -0.0062110, -0.0012986, 0.0016497
4: 0.0026242, 0.0033107, 0.0026277, 0.0038687, -0.0012445, 0.0006830
5: 0.0125822, 0.0176034, 0.0126044, 0.0229952, -0.0104129, 0.0049990
6: -0.0025456, -0.0016527, -0.0027121, -0.0016526, -0.0008929, 0.0010594
7: -0.0097238, -0.0074136, -0.0101546, -0.0074282, -0.0022956, 0.0027410
8: -0.0046778, -0.0034629, -0.0049044, -0.0017332, -0.0029446, 0.0014415
9: 0.0021515, 0.0035603, 0.0021604, 0.0038230, -0.0016715, 0.0013999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0062874
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0062935
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9783399, 0.9892431, -0.0041194, 0.0109081
1: -0.0044687, -0.0039431, -0.0045673, -0.0039262, -0.0005425, 0.0006242
2: 0.0108421, 0.0136275, 0.0108486, 0.0141502, -0.0033080, 0.0027789
3: -0.0075092, -0.0062080, -0.0078530, -0.0062110, -0.0012982, 0.0016450
4: 0.0026264, 0.0033099, 0.0026276, 0.0038691, -0.0012427, 0.0006823
5: 0.0125960, 0.0175961, 0.0126042, 0.0229984, -0.0104024, 0.0049919
6: -0.0025453, -0.0016562, -0.0027122, -0.0016522, -0.0008931, 0.0010560
7: -0.0097232, -0.0074226, -0.0101549, -0.0074280, -0.0022952, 0.0027322
8: -0.0046775, -0.0034676, -0.0049045, -0.0017319, -0.0029456, 0.0014369
9: 0.0021571, 0.0035599, 0.0021603, 0.0038231, -0.0016661, 0.0013996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063478, upper bound: 0.0062874
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063478, upper bound: 0.0062935
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9851146, 0.9892564, 0.9770324, 0.9894374, -0.0043228, 0.0122240
1: -0.0044688, -0.0039410, -0.0045863, -0.0038513, -0.0006175, 0.0006453
2: 0.0108312, 0.0136282, 0.0105921, 0.0142509, -0.0034197, 0.0030362
3: -0.0075097, -0.0062030, -0.0079192, -0.0060942, -0.0014155, 0.0017162
4: 0.0026242, 0.0033107, 0.0025780, 0.0039768, -0.0013526, 0.0007327
5: 0.0125822, 0.0176034, 0.0122815, 0.0240397, -0.0114575, 0.0053219
6: -0.0025456, -0.0016527, -0.0027443, -0.0015291, -0.0010165, 0.0010917
7: -0.0097238, -0.0074136, -0.0102381, -0.0072161, -0.0025077, 0.0028245
8: -0.0046778, -0.0034629, -0.0049483, -0.0013108, -0.0033670, 0.0014854
9: 0.0021515, 0.0035603, 0.0020311, 0.0038739, -0.0017224, 0.0015292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064389, upper bound: 0.0062874
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064389, upper bound: 0.0062935
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9851237, 0.9892480, 0.9770368, 0.9894375, -0.0043138, 0.0122111
1: -0.0044687, -0.0039431, -0.0045862, -0.0038515, -0.0006171, 0.0006432
2: 0.0108421, 0.0136275, 0.0105919, 0.0142506, -0.0034085, 0.0030356
3: -0.0075092, -0.0062080, -0.0079190, -0.0060941, -0.0014151, 0.0017110
4: 0.0026264, 0.0033099, 0.0025779, 0.0039765, -0.0013501, 0.0007320
5: 0.0125960, 0.0175961, 0.0122813, 0.0240362, -0.0114402, 0.0053149
6: -0.0025453, -0.0016562, -0.0027442, -0.0015295, -0.0010158, 0.0010881
7: -0.0097232, -0.0074226, -0.0102378, -0.0072160, -0.0025072, 0.0028152
8: -0.0046775, -0.0034676, -0.0049481, -0.0013122, -0.0033653, 0.0014805
9: 0.0021571, 0.0035599, 0.0020310, 0.0038737, -0.0017167, 0.0015289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064399, upper bound: 0.0062874
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0064399, upper bound: 0.0062935
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9783440, 0.9892430, -0.0050566, 0.0110841
1: -0.0044823, -0.0038982, -0.0045672, -0.0039264, -0.0005559, 0.0006690
2: 0.0106044, 0.0136997, 0.0108488, 0.0141499, -0.0035455, 0.0028509
3: -0.0075567, -0.0060998, -0.0078527, -0.0062110, -0.0013457, 0.0017530
4: 0.0025803, 0.0033872, 0.0026277, 0.0038687, -0.0012884, 0.0007595
5: 0.0122970, 0.0183425, 0.0126044, 0.0229952, -0.0106982, 0.0057381
6: -0.0025684, -0.0015803, -0.0027121, -0.0016526, -0.0009158, 0.0011318
7: -0.0097828, -0.0072263, -0.0101546, -0.0074282, -0.0023547, 0.0029283
8: -0.0047088, -0.0033644, -0.0049044, -0.0017332, -0.0029756, 0.0015400
9: 0.0020373, 0.0035963, 0.0021604, 0.0038230, -0.0017857, 0.0014359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063677
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063727
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9783399, 0.9892431, -0.0049243, 0.0110757
1: -0.0044804, -0.0039013, -0.0045673, -0.0039262, -0.0005542, 0.0006660
2: 0.0106209, 0.0136895, 0.0108486, 0.0141502, -0.0035292, 0.0028409
3: -0.0075500, -0.0061073, -0.0078530, -0.0062110, -0.0013390, 0.0017456
4: 0.0025836, 0.0033763, 0.0026276, 0.0038691, -0.0012855, 0.0007486
5: 0.0123178, 0.0182370, 0.0126042, 0.0229984, -0.0106806, 0.0056328
6: -0.0025651, -0.0015856, -0.0027122, -0.0016522, -0.0009129, 0.0011266
7: -0.0097744, -0.0072399, -0.0101549, -0.0074280, -0.0023464, 0.0029149
8: -0.0047044, -0.0033716, -0.0049045, -0.0017319, -0.0029725, 0.0015329
9: 0.0020456, 0.0035911, 0.0021603, 0.0038231, -0.0017775, 0.0014308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063677
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063727
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9841864, 0.9894281, 0.9770324, 0.9894374, -0.0052510, 0.0123957
1: -0.0044823, -0.0038982, -0.0045863, -0.0038513, -0.0006310, 0.0006881
2: 0.0106044, 0.0136997, 0.0105921, 0.0142509, -0.0036465, 0.0031077
3: -0.0075567, -0.0060998, -0.0079192, -0.0060942, -0.0014625, 0.0018194
4: 0.0025803, 0.0033872, 0.0025780, 0.0039768, -0.0013965, 0.0008092
5: 0.0122970, 0.0183425, 0.0122815, 0.0240397, -0.0117427, 0.0060610
6: -0.0025684, -0.0015803, -0.0027443, -0.0015291, -0.0010393, 0.0011641
7: -0.0097828, -0.0072263, -0.0102381, -0.0072161, -0.0025667, 0.0030118
8: -0.0047088, -0.0033644, -0.0049483, -0.0013108, -0.0033980, 0.0015839
9: 0.0020373, 0.0035963, 0.0020311, 0.0038739, -0.0018366, 0.0015652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063677
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063727
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9843188, 0.9894156, 0.9770368, 0.9894375, -0.0051187, 0.0123788
1: -0.0044804, -0.0039013, -0.0045862, -0.0038515, -0.0006288, 0.0006849
2: 0.0106209, 0.0136895, 0.0105919, 0.0142506, -0.0036296, 0.0030976
3: -0.0075500, -0.0061073, -0.0079190, -0.0060941, -0.0014559, 0.0018117
4: 0.0025836, 0.0033763, 0.0025779, 0.0039765, -0.0013929, 0.0007983
5: 0.0123178, 0.0182370, 0.0122813, 0.0240362, -0.0117184, 0.0059558
6: -0.0025651, -0.0015856, -0.0027442, -0.0015295, -0.0010356, 0.0011587
7: -0.0097744, -0.0072399, -0.0102378, -0.0072160, -0.0025585, 0.0029979
8: -0.0047044, -0.0033716, -0.0049481, -0.0013122, -0.0033922, 0.0015765
9: 0.0020456, 0.0035911, 0.0020310, 0.0038737, -0.0018281, 0.0015601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063677
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063727
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9796757, 0.9891557, -0.0056821, 0.0096816
1: -0.0044926, -0.0039158, -0.0045479, -0.0039661, -0.0005266, 0.0006320
2: 0.0106979, 0.0137547, 0.0109641, 0.0140473, -0.0033494, 0.0027906
3: -0.0075928, -0.0061423, -0.0077853, -0.0062635, -0.0013293, 0.0016429
4: 0.0025984, 0.0034459, 0.0026500, 0.0037590, -0.0011605, 0.0007960
5: 0.0124146, 0.0189102, 0.0127494, 0.0219347, -0.0095201, 0.0061608
6: -0.0025859, -0.0016101, -0.0026793, -0.0016951, -0.0008908, 0.0010692
7: -0.0098282, -0.0073035, -0.0100699, -0.0075233, -0.0023049, 0.0027664
8: -0.0047327, -0.0033853, -0.0048598, -0.0021621, -0.0025706, 0.0014745
9: 0.0020844, 0.0036239, 0.0022185, 0.0037713, -0.0016869, 0.0014055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063366
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063399
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9836155, 0.9893424, 0.9796717, 0.9891558, -0.0055404, 0.0096707
1: -0.0044906, -0.0039195, -0.0045479, -0.0039660, -0.0005245, 0.0006284
2: 0.0107175, 0.0137437, 0.0109639, 0.0140476, -0.0033301, 0.0027799
3: -0.0075856, -0.0061513, -0.0077855, -0.0062634, -0.0013222, 0.0016342
4: 0.0026022, 0.0034343, 0.0026499, 0.0037593, -0.0011571, 0.0007843
5: 0.0124392, 0.0187972, 0.0127491, 0.0219379, -0.0094987, 0.0060481
6: -0.0025824, -0.0016164, -0.0026794, -0.0016950, -0.0008874, 0.0010631
7: -0.0098192, -0.0073197, -0.0100701, -0.0075232, -0.0022960, 0.0027504
8: -0.0047280, -0.0034135, -0.0048599, -0.0021608, -0.0025671, 0.0014464
9: 0.0020943, 0.0036184, 0.0022184, 0.0037715, -0.0016772, 0.0014001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063366
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063399
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9786741, 0.9893333, -0.0058597, 0.0106832
1: -0.0044926, -0.0039158, -0.0045624, -0.0039218, -0.0005708, 0.0006466
2: 0.0106979, 0.0137547, 0.0107296, 0.0141244, -0.0034266, 0.0030251
3: -0.0075928, -0.0061423, -0.0078360, -0.0061568, -0.0014360, 0.0016937
4: 0.0025984, 0.0034459, 0.0026046, 0.0038415, -0.0012431, 0.0008414
5: 0.0124146, 0.0189102, 0.0124544, 0.0227323, -0.0103178, 0.0064557
6: -0.0025859, -0.0016101, -0.0027040, -0.0016202, -0.0009657, 0.0010938
7: -0.0098282, -0.0073035, -0.0101336, -0.0073297, -0.0024985, 0.0028301
8: -0.0047327, -0.0033853, -0.0048933, -0.0018395, -0.0028932, 0.0015081
9: 0.0020844, 0.0036239, 0.0021004, 0.0038102, -0.0017258, 0.0015236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063357
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063396
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9836155, 0.9893424, 0.9786724, 0.9893335, -0.0057181, 0.0106700
1: -0.0044906, -0.0039195, -0.0045624, -0.0039218, -0.0005688, 0.0006429
2: 0.0107175, 0.0137437, 0.0107294, 0.0141246, -0.0034071, 0.0030143
3: -0.0075856, -0.0061513, -0.0078361, -0.0061567, -0.0014289, 0.0016848
4: 0.0026022, 0.0034343, 0.0026045, 0.0038417, -0.0012394, 0.0008297
5: 0.0124392, 0.0187972, 0.0124542, 0.0227336, -0.0102944, 0.0063430
6: -0.0025824, -0.0016164, -0.0027040, -0.0016202, -0.0009623, 0.0010876
7: -0.0098192, -0.0073197, -0.0101337, -0.0073295, -0.0024896, 0.0028140
8: -0.0047280, -0.0034135, -0.0048934, -0.0018390, -0.0028890, 0.0014799
9: 0.0020943, 0.0036184, 0.0021003, 0.0038102, -0.0017160, 0.0015182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063940, upper bound: 0.0063357
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063940, upper bound: 0.0063396
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9823954, 0.9895465, 0.9796757, 0.9891557, -0.0067603, 0.0098708
1: -0.0045083, -0.0038687, -0.0045479, -0.0039661, -0.0005422, 0.0006792
2: 0.0104480, 0.0138377, 0.0109641, 0.0140473, -0.0035992, 0.0028737
3: -0.0076474, -0.0060286, -0.0077853, -0.0062635, -0.0013840, 0.0017567
4: 0.0025501, 0.0035348, 0.0026500, 0.0037590, -0.0012089, 0.0008848
5: 0.0121003, 0.0197688, 0.0127494, 0.0219347, -0.0098344, 0.0070194
6: -0.0026124, -0.0015304, -0.0026793, -0.0016951, -0.0009173, 0.0011490
7: -0.0098968, -0.0070971, -0.0100699, -0.0075233, -0.0023735, 0.0029728
8: -0.0047688, -0.0030380, -0.0048598, -0.0021621, -0.0026067, 0.0018218
9: 0.0019586, 0.0036658, 0.0022185, 0.0037713, -0.0018128, 0.0014473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064237
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064255
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9826351, 0.9895267, 0.9796717, 0.9891558, -0.0065207, 0.0098551
1: -0.0045048, -0.0038736, -0.0045479, -0.0039660, -0.0005388, 0.0006743
2: 0.0104740, 0.0138193, 0.0109639, 0.0140476, -0.0035736, 0.0028554
3: -0.0076353, -0.0060404, -0.0077855, -0.0062634, -0.0013719, 0.0017450
4: 0.0025551, 0.0035150, 0.0026499, 0.0037593, -0.0012042, 0.0008651
5: 0.0121330, 0.0195778, 0.0127491, 0.0219379, -0.0098049, 0.0068287
6: -0.0026065, -0.0015387, -0.0026794, -0.0016950, -0.0009115, 0.0011408
7: -0.0098816, -0.0071186, -0.0100701, -0.0075232, -0.0023584, 0.0029515
8: -0.0047608, -0.0031152, -0.0048599, -0.0021608, -0.0025999, 0.0017447
9: 0.0019716, 0.0036565, 0.0022184, 0.0037715, -0.0017998, 0.0014381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064240
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064270
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9823954, 0.9895465, 0.9786741, 0.9893333, -0.0069379, 0.0108724
1: -0.0045083, -0.0038687, -0.0045624, -0.0039218, -0.0005865, 0.0006937
2: 0.0104480, 0.0138377, 0.0107296, 0.0141244, -0.0036764, 0.0031082
3: -0.0076474, -0.0060286, -0.0078360, -0.0061568, -0.0014907, 0.0018074
4: 0.0025501, 0.0035348, 0.0026046, 0.0038415, -0.0012915, 0.0009302
5: 0.0121003, 0.0197688, 0.0124544, 0.0227323, -0.0106320, 0.0073144
6: -0.0026124, -0.0015304, -0.0027040, -0.0016202, -0.0009922, 0.0011736
7: -0.0098968, -0.0070971, -0.0101336, -0.0073297, -0.0025671, 0.0030365
8: -0.0047688, -0.0030380, -0.0048933, -0.0018395, -0.0029293, 0.0018553
9: 0.0019586, 0.0036658, 0.0021004, 0.0038102, -0.0018516, 0.0015654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064237
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064255
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9826351, 0.9895267, 0.9786724, 0.9893335, -0.0066984, 0.0108544
1: -0.0045048, -0.0038736, -0.0045624, -0.0039218, -0.0005830, 0.0006889
2: 0.0104740, 0.0138193, 0.0107294, 0.0141246, -0.0036506, 0.0030898
3: -0.0076353, -0.0060404, -0.0078361, -0.0061567, -0.0014786, 0.0017957
4: 0.0025551, 0.0035150, 0.0026045, 0.0038417, -0.0012866, 0.0009105
5: 0.0121330, 0.0195778, 0.0124542, 0.0227336, -0.0106006, 0.0071236
6: -0.0026065, -0.0015387, -0.0027040, -0.0016202, -0.0009864, 0.0011654
7: -0.0098816, -0.0071186, -0.0101337, -0.0073295, -0.0025520, 0.0030151
8: -0.0047608, -0.0031152, -0.0048934, -0.0018390, -0.0029218, 0.0017781
9: 0.0019716, 0.0036565, 0.0021003, 0.0038102, -0.0018386, 0.0015562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064240
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064270
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9783440, 0.9892430, -0.0057694, 0.0110133
1: -0.0044926, -0.0039158, -0.0045672, -0.0039264, -0.0005662, 0.0006514
2: 0.0106979, 0.0137547, 0.0108488, 0.0141499, -0.0034520, 0.0029059
3: -0.0075928, -0.0061423, -0.0078527, -0.0062110, -0.0013818, 0.0017104
4: 0.0025984, 0.0034459, 0.0026277, 0.0038687, -0.0012703, 0.0008183
5: 0.0124146, 0.0189102, 0.0126044, 0.0229952, -0.0105806, 0.0063058
6: -0.0025859, -0.0016101, -0.0027121, -0.0016526, -0.0009333, 0.0011020
7: -0.0098282, -0.0073035, -0.0101546, -0.0074282, -0.0024000, 0.0028511
8: -0.0047327, -0.0033853, -0.0049044, -0.0017332, -0.0029995, 0.0015191
9: 0.0020844, 0.0036239, 0.0021604, 0.0038230, -0.0017386, 0.0014635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063290
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063341
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9836155, 0.9893424, 0.9783399, 0.9892431, -0.0056276, 0.0110025
1: -0.0044906, -0.0039195, -0.0045673, -0.0039262, -0.0005644, 0.0006477
2: 0.0107175, 0.0137437, 0.0108486, 0.0141502, -0.0034327, 0.0028951
3: -0.0075856, -0.0061513, -0.0078530, -0.0062110, -0.0013747, 0.0017017
4: 0.0026022, 0.0034343, 0.0026276, 0.0038691, -0.0012668, 0.0008066
5: 0.0124392, 0.0187972, 0.0126042, 0.0229984, -0.0105592, 0.0061930
6: -0.0025824, -0.0016164, -0.0027122, -0.0016522, -0.0009302, 0.0010958
7: -0.0098192, -0.0073197, -0.0101549, -0.0074280, -0.0023912, 0.0028352
8: -0.0047280, -0.0034135, -0.0049045, -0.0017319, -0.0029960, 0.0014910
9: 0.0020943, 0.0036184, 0.0021603, 0.0038231, -0.0017289, 0.0014581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063290
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063341
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9834736, 0.9893573, 0.9770324, 0.9894374, -0.0059638, 0.0123249
1: -0.0044926, -0.0039158, -0.0045863, -0.0038513, -0.0006414, 0.0006704
2: 0.0106979, 0.0137547, 0.0105921, 0.0142509, -0.0035530, 0.0031626
3: -0.0075928, -0.0061423, -0.0079192, -0.0060942, -0.0014986, 0.0017769
4: 0.0025984, 0.0034459, 0.0025780, 0.0039768, -0.0013784, 0.0008680
5: 0.0124146, 0.0189102, 0.0122815, 0.0240397, -0.0116251, 0.0066287
6: -0.0025859, -0.0016101, -0.0027443, -0.0015291, -0.0010568, 0.0011342
7: -0.0098282, -0.0073035, -0.0102381, -0.0072161, -0.0026121, 0.0029346
8: -0.0047327, -0.0033853, -0.0049483, -0.0013108, -0.0034219, 0.0015630
9: 0.0020844, 0.0036239, 0.0020311, 0.0038739, -0.0017895, 0.0015928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063286
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063341
time: 0.89 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063933, upper bound: 0.0063933
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063933, upper bound: 0.0063971
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063933
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063971, upper bound: 0.0063971
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064708, upper bound: 0.0063920
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064708, upper bound: 0.0063966
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064731, upper bound: 0.0063920
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064731, upper bound: 0.0063966
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064708
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064731
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064708
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064708
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063920, upper bound: 0.0064731
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064708
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063966, upper bound: 0.0064738
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064048, upper bound: 0.0063544
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064048, upper bound: 0.0063646
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064098, upper bound: 0.0063544
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064098, upper bound: 0.0063652
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064860, upper bound: 0.0063536
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064860, upper bound: 0.0063634
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064881, upper bound: 0.0063536
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064881, upper bound: 0.0063642
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064292
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064393
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064093, upper bound: 0.0064292
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064093, upper bound: 0.0064416
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064292
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064035, upper bound: 0.0064393
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063951, upper bound: 0.0064292
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064093, upper bound: 0.0064416
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0064048
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0064098
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0064048
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0064115
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0064035
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0064093
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0064035
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0064109
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064860
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064881
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064866
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064929
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064860
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064881
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064866
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064929
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0063962
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063544, upper bound: 0.0064030
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0063962
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063646, upper bound: 0.0064059
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0063948
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064292, upper bound: 0.0064025
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0063948
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064393, upper bound: 0.0064052
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064760
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064803
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064766
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064875
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064760
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063536, upper bound: 0.0064803
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064766
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063634, upper bound: 0.0064875
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063432, upper bound: 0.0063257
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063432, upper bound: 0.0063299
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0063257
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0063299
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064274, upper bound: 0.0063247
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064274, upper bound: 0.0063296
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064300, upper bound: 0.0063247
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064300, upper bound: 0.0063296
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064082
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064110
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064082
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064116
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064082
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063426, upper bound: 0.0064110
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064082
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063463, upper bound: 0.0064116
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0062874
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063464, upper bound: 0.0062935
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063478, upper bound: 0.0062874
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063478, upper bound: 0.0062935
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064389, upper bound: 0.0062874
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064389, upper bound: 0.0062935
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064399, upper bound: 0.0062874
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0064399, upper bound: 0.0062935
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063677
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063727
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063677
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063727
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063677
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063456, upper bound: 0.0063727
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063677
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063474, upper bound: 0.0063727
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063366
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063399
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063366
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063399
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063357
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063396
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063940, upper bound: 0.0063357
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063940, upper bound: 0.0063396
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064237
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064255
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064240
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064270
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064237
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063055, upper bound: 0.0064255
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064240
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0064270
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063290
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063058, upper bound: 0.0063341
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063290
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063104, upper bound: 0.0063341
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063286
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0063881, upper bound: 0.0063341
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0063341
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064197
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064215
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063152, upper bound: 0.0064197
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063112, upper bound: 0.0064215
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063336, upper bound: 0.0063463
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063299, upper bound: 0.0063467
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063336, upper bound: 0.0063463
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063299, upper bound: 0.0063467
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063334, upper bound: 0.0064300
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063296, upper bound: 0.0064314
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063334, upper bound: 0.0064300
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063296, upper bound: 0.0064314
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063460, upper bound: 0.0063104
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063399, upper bound: 0.0063112
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063460, upper bound: 0.0063104
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063399, upper bound: 0.0063112
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063458, upper bound: 0.0063940
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063396, upper bound: 0.0063944
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063458, upper bound: 0.0063940
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063257, upper bound: 0.0063944
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063474
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063507
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063474
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063507
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064399
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064428
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064399
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064428
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063403
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063430
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063403
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0063430
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064336
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064364
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064336
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062935, upper bound: 0.0064364
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063200, upper bound: 0.0063224
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0063224
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063200, upper bound: 0.0063224
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0063224
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063198, upper bound: 0.0064124
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0064126
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063198, upper bound: 0.0064124
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063095, upper bound: 0.0064126
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063281, upper bound: 0.0062909
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0062910
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063281, upper bound: 0.0062909
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0062910
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063278, upper bound: 0.0063775
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0063775
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063278, upper bound: 0.0063775
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0063189, upper bound: 0.0063774
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063280
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063288
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063280
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063288
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064253
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064259
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064253
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064259
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063225
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063244
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0063225
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0063244
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064191
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064204
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062794, upper bound: 0.0064191
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -0.0062786, upper bound: 0.0064204

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.09 + 598.31 = 601.41 seconds
