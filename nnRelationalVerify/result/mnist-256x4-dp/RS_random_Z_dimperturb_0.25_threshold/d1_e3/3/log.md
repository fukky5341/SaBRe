## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00603328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0009419, 0.0009419)
1: (0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0028830, 0.0028830)
2: (0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0021477, 0.0021477)
3: (0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0031277, 0.0031277)
4: (-0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0033288, 0.0033288)
5: (0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0037430, 0.0037430)
6: (0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0030778, 0.0030778)
7: (-0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0029265, 0.0029265)
8: (0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0029209, 0.0029209)
9: (0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0136595, 0.0136595)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.34 = 2.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0072556, upper bound: 0.0072556

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062357, upper bound: 0.0066147
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0066147, upper bound: 0.0062357
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.04
Output dim: 9, lower bound: -0.0062357, upper bound: 0.0066147
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.04
Output dim: 9, lower bound: -0.0066147, upper bound: 0.0062357

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0008992, 0.0008867
1: 0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0025575, 0.0024475
2: 0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0019313, 0.0018480
3: 0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0028645, 0.0027665
4: -0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0029494, 0.0030628
5: 0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0034357, 0.0033110
6: 0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0028276, 0.0027299
7: -0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0025750, 0.0026776
8: 0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0026738, 0.0025866
9: 0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0120117, 0.0124523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060096, upper bound: 0.0062872
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059959, upper bound: 0.0064092
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0008867, 0.0008992
1: 0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0024475, 0.0025575
2: 0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0018480, 0.0019313
3: 0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0027665, 0.0028645
4: -0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0030628, 0.0029494
5: 0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0033110, 0.0034357
6: 0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0027299, 0.0028276
7: -0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0026776, 0.0025750
8: 0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0025866, 0.0026738
9: 0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0124523, 0.0120117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0064092, upper bound: 0.0059959
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062872, upper bound: 0.0060096
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 9, lower bound: -0.0060096, upper bound: 0.0062872
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 9, lower bound: -0.0059959, upper bound: 0.0064092
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 9, lower bound: -0.0064092, upper bound: 0.0059959
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 9, lower bound: -0.0062872, upper bound: 0.0060096

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0008966, 0.0008838
1: 0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0024298, 0.0023283
2: 0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0018768, 0.0017942
3: 0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0027663, 0.0026723
4: -0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0028826, 0.0029961
5: 0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0033460, 0.0032235
6: 0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0027546, 0.0026603
7: -0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0025180, 0.0026213
8: 0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0025730, 0.0024899
9: 0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0116151, 0.0120435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0055686, upper bound: 0.0053405
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0051965, upper bound: 0.0059024
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0008965, 0.0008842
1: 0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0024495, 0.0023197
2: 0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0018855, 0.0017935
3: 0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0027785, 0.0026683
4: -0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0028827, 0.0030028
5: 0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0033562, 0.0032213
6: 0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0027627, 0.0026570
7: -0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0025187, 0.0026278
8: 0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0025848, 0.0024857
9: 0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0116030, 0.0121023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056840, upper bound: 0.0061599
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0057357, upper bound: 0.0060030
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0008842, 0.0008965
1: 0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0023197, 0.0024495
2: 0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0017935, 0.0018855
3: 0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0026683, 0.0027785
4: -0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0030028, 0.0028827
5: 0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0032213, 0.0033562
6: 0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0026570, 0.0027627
7: -0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0026278, 0.0025187
8: 0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0024857, 0.0025848
9: 0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0121023, 0.0116030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0057406, upper bound: 0.0055774
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0059969, upper bound: 0.0054120
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0008838, 0.0008966
1: 0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0023283, 0.0024298
2: 0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0017942, 0.0018768
3: 0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0026723, 0.0027663
4: -0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0029961, 0.0028826
5: 0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0032235, 0.0033460
6: 0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0026603, 0.0027546
7: -0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0026213, 0.0025180
8: 0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0024899, 0.0025730
9: 0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0120435, 0.0116151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0056559, upper bound: 0.0055913
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0058734, upper bound: 0.0054348
time: 0.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0055686, upper bound: 0.0053405
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0051965, upper bound: 0.0059024
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0056840, upper bound: 0.0061599
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0057357, upper bound: 0.0060030
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0057406, upper bound: 0.0055774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0059969, upper bound: 0.0054120
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0056559, upper bound: 0.0055913
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 9, lower bound: -0.0058734, upper bound: 0.0054348

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0030748, -0.0019434, -0.0030748, -0.0019434, -0.0007734, 0.0007526
1: 0.0227173, 0.0284588, 0.0227173, 0.0284588, -0.0022121, 0.0020553
2: 0.0226192, 0.0264664, 0.0226192, 0.0264664, -0.0016158, 0.0014878
3: 0.0101968, 0.0145602, 0.0101968, 0.0145602, -0.0023693, 0.0022102
4: -0.0148971, -0.0103870, -0.0148971, -0.0103870, -0.0022980, 0.0024870
5: 0.0174203, 0.0226437, 0.0174203, 0.0226437, -0.0027817, 0.0025787
6: 0.0082406, 0.0123566, 0.0082406, 0.0123566, -0.0023086, 0.0021465
7: -0.0196229, -0.0154230, -0.0196229, -0.0154230, -0.0020395, 0.0022201
8: 0.0122132, 0.0163825, 0.0122132, 0.0163825, -0.0021871, 0.0020484
9: 0.9135485, 0.9338936, 0.9135485, 0.9338936, -0.0094811, 0.0102197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0052499, upper bound: 0.0051418
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049073, upper bound: 0.0057778
time: 0.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.00
Output dim: 9, lower bound: -0.0052499, upper bound: 0.0051418
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.00
Output dim: 9, lower bound: -0.0049073, upper bound: 0.0057778

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.63 + 18.06 = 20.70 seconds
