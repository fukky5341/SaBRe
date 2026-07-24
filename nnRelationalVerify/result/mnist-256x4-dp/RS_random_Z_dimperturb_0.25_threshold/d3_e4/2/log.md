## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0024309


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002870, 0.0002870)
1: (-0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015888, 0.0015888)
2: (0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0035497, 0.0035497)
3: (0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014958, 0.0014958)
4: (0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0058033, 0.0058033)
5: (0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011290, 0.0011290)
6: (-0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014692, 0.0014692)
7: (-0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001874, 0.0001874)
8: (-0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0010151, 0.0010151)
9: (-0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0050818, 0.0050818)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.92 + 1.55 = 3.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0037932, upper bound: 0.0037932

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032373, upper bound: 0.0032373
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032373, upper bound: 0.0032373
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -0.0032373, upper bound: 0.0032373
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -0.0032373, upper bound: 0.0032373

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002864, 0.0002868
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015879, 0.0015857
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0035427, 0.0035476
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014950, 0.0014929
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0057999, 0.0057920
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011283, 0.0011268
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014663, 0.0014683
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001870, 0.0001873
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0010145, 0.0010131
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0050719, 0.0050788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032090, upper bound: 0.0031800
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031800, upper bound: 0.0032090
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002868, 0.0002870
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015888, 0.0015879
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0035476, 0.0035497
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014958, 0.0014950
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0058033, 0.0057999
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011290, 0.0011283
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014683, 0.0014692
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001873, 0.0001874
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0010151, 0.0010145
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0050788, 0.0050818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032173, upper bound: 0.0032172
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0032172, upper bound: 0.0032173
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -0.0032090, upper bound: 0.0031800
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -0.0031800, upper bound: 0.0032090
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -0.0032173, upper bound: 0.0032172
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -0.0032172, upper bound: 0.0032173

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002811, 0.0002822
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015626, 0.0015563
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034770, 0.0034910
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014711, 0.0014652
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0057073, 0.0056844
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011103, 0.0011058
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014391, 0.0014449
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001836, 0.0001843
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009983, 0.0009943
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049777, 0.0049978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030751, upper bound: 0.0030479
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030751, upper bound: 0.0030481
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002818, 0.0002815
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015585, 0.0015603
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034859, 0.0034818
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014672, 0.0014689
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0056924, 0.0056990
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011074, 0.0011087
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014428, 0.0014411
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001840, 0.0001838
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009957, 0.0009968
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049904, 0.0049847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031596, upper bound: 0.0031888
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031579, upper bound: 0.0031890
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002835, 0.0002838
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015712, 0.0015698
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0035072, 0.0035102
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014792, 0.0014779
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0057388, 0.0057338
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011164, 0.0011155
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014516, 0.0014529
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001852, 0.0001853
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0010038, 0.0010029
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0050210, 0.0050253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031890, upper bound: 0.0031579
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031596, upper bound: 0.0031888
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002836, 0.0002837
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015708, 0.0015703
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0035081, 0.0035093
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014788, 0.0014783
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0057372, 0.0057354
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011161, 0.0011158
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014520, 0.0014525
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001852, 0.0001853
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0010035, 0.0010032
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0050223, 0.0050240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020433, upper bound: 0.0020365
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020433, upper bound: 0.0020365
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0030751, upper bound: 0.0030479
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0030751, upper bound: 0.0030481
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0031596, upper bound: 0.0031888
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0031579, upper bound: 0.0031890
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0031890, upper bound: 0.0031579
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0031596, upper bound: 0.0031888
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0020433, upper bound: 0.0020365
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -0.0020433, upper bound: 0.0020365

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002796, 0.0002810
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015561, 0.0015483
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034590, 0.0034766
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014650, 0.0014576
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0056838, 0.0056550
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011057, 0.0011001
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014317, 0.0014389
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001826, 0.0001836
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009942, 0.0009892
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049520, 0.0049772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030041, upper bound: 0.0029999
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030260, upper bound: 0.0029796
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002799, 0.0002808
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015545, 0.0015498
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034624, 0.0034730
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014635, 0.0014591
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0056780, 0.0056606
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0011046, 0.0011012
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014331, 0.0014375
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001828, 0.0001834
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009932, 0.0009901
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049569, 0.0049721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030035, upper bound: 0.0030003
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030260, upper bound: 0.0029796
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002784, 0.0002783
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015407, 0.0015418
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034445, 0.0034421
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014505, 0.0014515
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0056273, 0.0056313
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010947, 0.0010955
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014256, 0.0014246
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001819, 0.0001817
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009843, 0.0009850
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049312, 0.0049277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030979, upper bound: 0.0031362
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0031066, upper bound: 0.0031224
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002786, 0.0002782
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015403, 0.0015425
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034461, 0.0034411
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014501, 0.0014522
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0056258, 0.0056339
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010944, 0.0010960
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014263, 0.0014243
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001819, 0.0001817
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009840, 0.0009855
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049335, 0.0049264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030611, upper bound: 0.0030896
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030629, upper bound: 0.0030895
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002782, 0.0002792
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015457, 0.0015403
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034411, 0.0034532
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014552, 0.0014501
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0056455, 0.0056258
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010983, 0.0010944
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014243, 0.0014292
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001817, 0.0001823
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009875, 0.0009840
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049264, 0.0049436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020074, upper bound: 0.0019989
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0020074, upper bound: 0.0019989
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002789, 0.0002784
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015416, 0.0015440
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0034494, 0.0034440
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014513, 0.0014536
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0056305, 0.0056394
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010954, 0.0010971
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0014277, 0.0014255
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001821, 0.0001818
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009849, 0.0009864
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0049383, 0.0049305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030746, upper bound: 0.0030897
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030639, upper bound: 0.0031054
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030041, upper bound: 0.0029999
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030260, upper bound: 0.0029796
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030035, upper bound: 0.0030003
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030260, upper bound: 0.0029796
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030979, upper bound: 0.0031362
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0031066, upper bound: 0.0031224
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030611, upper bound: 0.0030896
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030629, upper bound: 0.0030895
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0020074, upper bound: 0.0019989
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0020074, upper bound: 0.0019989
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030746, upper bound: 0.0030897
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 4, lower bound: -0.0030639, upper bound: 0.0031054

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002728, 0.0002745
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015200, 0.0015104
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033745, 0.0033959
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014310, 0.0014220
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0055519, 0.0055169
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010801, 0.0010732
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013967, 0.0014056
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001782, 0.0001793
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009711, 0.0009650
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0048310, 0.0048617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029138, upper bound: 0.0028949
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029008, upper bound: 0.0029087
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002731, 0.0002742
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015183, 0.0015122
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033783, 0.0033920
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014294, 0.0014236
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0055454, 0.0055231
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010788, 0.0010745
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013983, 0.0014039
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001784, 0.0001791
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009700, 0.0009661
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0048365, 0.0048560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030036, upper bound: 0.0029562
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0030027, upper bound: 0.0029565
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002730, 0.0002742
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015184, 0.0015118
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033774, 0.0033923
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014295, 0.0014233
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0055461, 0.0055217
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010789, 0.0010742
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013979, 0.0014041
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001783, 0.0001791
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009701, 0.0009658
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0048352, 0.0048566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029812, upper bound: 0.0029761
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029811, upper bound: 0.0029774
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002734, 0.0002739
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015166, 0.0015137
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033817, 0.0033882
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014278, 0.0014251
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0055394, 0.0055287
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010776, 0.0010756
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013997, 0.0014024
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001785, 0.0001789
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009689, 0.0009671
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0048414, 0.0048507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029346, upper bound: 0.0028778
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029234, upper bound: 0.0028884
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002718, 0.0002718
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015049, 0.0015049
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033620, 0.0033622
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014168, 0.0014168
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054968, 0.0054965
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010694, 0.0010693
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013915, 0.0013916
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001775, 0.0001775
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009615, 0.0009614
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0048131, 0.0048134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019341, upper bound: 0.0019618
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019341, upper bound: 0.0019618
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002720, 0.0002715
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015032, 0.0015060
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033646, 0.0033583
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014152, 0.0014179
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054905, 0.0055008
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010681, 0.0010701
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013926, 0.0013900
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001776, 0.0001773
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009604, 0.0009622
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0048169, 0.0048079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029774, upper bound: 0.0029811
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029771, upper bound: 0.0029816
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002587, 0.0002596
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014374, 0.0014325
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0032003, 0.0032113
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013533, 0.0013486
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0052501, 0.0052320
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010214, 0.0010178
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013246, 0.0013291
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001690, 0.0001695
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009183, 0.0009152
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0045816, 0.0045974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029759, upper bound: 0.0029876
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029638, upper bound: 0.0030052
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002600, 0.0002585
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014313, 0.0014396
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0032163, 0.0031976
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013475, 0.0013554
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0052278, 0.0052583
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010170, 0.0010229
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013312, 0.0013235
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001698, 0.0001688
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009144, 0.0009198
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0046045, 0.0045778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029768, upper bound: 0.0029877
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029638, upper bound: 0.0030052
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002589, 0.0002586
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014320, 0.0014335
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0032026, 0.0031992
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013481, 0.0013496
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0052303, 0.0052358
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010175, 0.0010186
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013255, 0.0013241
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001691, 0.0001689
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009149, 0.0009158
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0045849, 0.0045800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019116, upper bound: 0.0019305
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019116, upper bound: 0.0019305
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002590, 0.0002585
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014312, 0.0014340
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0032037, 0.0031974
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013474, 0.0013500
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0052273, 0.0052376
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010169, 0.0010189
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013260, 0.0013234
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001691, 0.0001688
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009143, 0.0009161
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0045864, 0.0045774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019098, upper bound: 0.0019307
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0019098, upper bound: 0.0019307
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029138, upper bound: 0.0028949
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029008, upper bound: 0.0029087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0030036, upper bound: 0.0029562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0030027, upper bound: 0.0029565
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029812, upper bound: 0.0029761
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029811, upper bound: 0.0029774
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029346, upper bound: 0.0028778
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029234, upper bound: 0.0028884
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0019341, upper bound: 0.0019618
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0019341, upper bound: 0.0019618
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029774, upper bound: 0.0029811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029771, upper bound: 0.0029816
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029759, upper bound: 0.0029876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029638, upper bound: 0.0030052
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029768, upper bound: 0.0029877
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0029638, upper bound: 0.0030052
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0019116, upper bound: 0.0019305
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0019116, upper bound: 0.0019305
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0019098, upper bound: 0.0019307
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 4, lower bound: -0.0019098, upper bound: 0.0019307

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002544, 0.0002562
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014183, 0.0014085
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0031467, 0.0031687
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013353, 0.0013260
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0051804, 0.0051445
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010078, 0.0010008
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013024, 0.0013115
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001661, 0.0001673
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009061, 0.0008998
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0045049, 0.0045364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028174, upper bound: 0.0028057
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028205, upper bound: 0.0027980
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002546, 0.0002561
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014181, 0.0014097
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0031495, 0.0031681
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013351, 0.0013272
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0051795, 0.0051490
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010076, 0.0010017
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013035, 0.0013113
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001663, 0.0001673
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009060, 0.0009006
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0045089, 0.0045356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028759, upper bound: 0.0028827
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028760, upper bound: 0.0028833
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002699, 0.0002711
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015010, 0.0014946
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033391, 0.0033534
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014131, 0.0014071
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054825, 0.0054590
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010666, 0.0010620
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013820, 0.0013880
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001763, 0.0001770
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009590, 0.0009549
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0047803, 0.0048009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029095, upper bound: 0.0028522
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028987, upper bound: 0.0028620
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002700, 0.0002709
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015002, 0.0014949
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033398, 0.0033516
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014124, 0.0014074
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054794, 0.0054602
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010660, 0.0010622
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013823, 0.0013872
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001763, 0.0001769
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009584, 0.0009551
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0047813, 0.0047982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029081, upper bound: 0.0028523
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028988, upper bound: 0.0028628
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002699, 0.0002711
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015012, 0.0014943
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033384, 0.0033538
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014133, 0.0014068
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054831, 0.0054578
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010667, 0.0010618
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013817, 0.0013881
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001763, 0.0001771
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009591, 0.0009547
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0047793, 0.0048014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028875, upper bound: 0.0028706
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028753, upper bound: 0.0028827
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002699, 0.0002710
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0015004, 0.0014945
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033389, 0.0033520
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014126, 0.0014070
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054802, 0.0054587
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010661, 0.0010619
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013820, 0.0013874
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001763, 0.0001770
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009586, 0.0009548
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0047801, 0.0047989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028872, upper bound: 0.0028711
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028755, upper bound: 0.0028839
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002550, 0.0002556
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014154, 0.0014117
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0031539, 0.0031621
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013325, 0.0013291
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0051697, 0.0051563
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010057, 0.0010031
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013054, 0.0013088
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001665, 0.0001669
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009043, 0.0009019
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0045153, 0.0045270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029095, upper bound: 0.0028527
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029081, upper bound: 0.0028527
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002551, 0.0002555
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014146, 0.0014123
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0031553, 0.0031604
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013318, 0.0013297
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0051669, 0.0051586
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010052, 0.0010035
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013060, 0.0013081
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001666, 0.0001669
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009038, 0.0009023
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0045172, 0.0045246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028271, upper bound: 0.0027961
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028282, upper bound: 0.0027917
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002706, 0.0002703
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014968, 0.0014982
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033471, 0.0033439
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014091, 0.0014105
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054669, 0.0054721
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010635, 0.0010645
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013853, 0.0013840
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001767, 0.0001765
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009563, 0.0009572
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0047917, 0.0047873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028839, upper bound: 0.0028755
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028711, upper bound: 0.0028872
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002708, 0.0002701
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0014953, 0.0014997
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0033504, 0.0033408
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0014078, 0.0014119
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0054618, 0.0054775
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0010625, 0.0010656
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0013867, 0.0013827
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001769, 0.0001764
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0009554, 0.0009581
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0047966, 0.0047827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028833, upper bound: 0.0028760
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028702, upper bound: 0.0028877
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002404, 0.0002413
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013362, 0.0013312
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029741, 0.0029851
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012579, 0.0012533
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048803, 0.0048624
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009494, 0.0009459
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012310, 0.0012355
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001570, 0.0001576
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008536, 0.0008505
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042579, 0.0042736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029117, upper bound: 0.0029351
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0029243, upper bound: 0.0029227
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002412, 0.0002413
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013362, 0.0013356
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029838, 0.0029852
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012580, 0.0012574
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048804, 0.0048781
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009494, 0.0009490
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012350, 0.0012356
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001575, 0.0001576
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008537, 0.0008533
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042717, 0.0042737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018166, upper bound: 0.0018279
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018166, upper bound: 0.0018279
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002417, 0.0002406
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013322, 0.0013384
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029902, 0.0029763
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012542, 0.0012601
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048659, 0.0048886
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009466, 0.0009510
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012376, 0.0012319
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001579, 0.0001571
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008511, 0.0008551
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042808, 0.0042609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028431, upper bound: 0.0028511
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028431, upper bound: 0.0028511
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002421, 0.0002402
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013301, 0.0013403
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029943, 0.0029715
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012522, 0.0012618
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048581, 0.0048954
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009451, 0.0009523
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012393, 0.0012299
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001581, 0.0001569
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008498, 0.0008563
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042868, 0.0042541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018174, upper bound: 0.0018255
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018174, upper bound: 0.0018255
time: 0.56 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028174, upper bound: 0.0028057
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028205, upper bound: 0.0027980
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028759, upper bound: 0.0028827
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028760, upper bound: 0.0028833
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0029095, upper bound: 0.0028522
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028987, upper bound: 0.0028620
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0029081, upper bound: 0.0028523
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028988, upper bound: 0.0028628
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028875, upper bound: 0.0028706
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028753, upper bound: 0.0028827
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028872, upper bound: 0.0028711
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028755, upper bound: 0.0028839
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0029095, upper bound: 0.0028527
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0029081, upper bound: 0.0028527
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028271, upper bound: 0.0027961
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028282, upper bound: 0.0027917
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028839, upper bound: 0.0028755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028711, upper bound: 0.0028872
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028833, upper bound: 0.0028760
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028702, upper bound: 0.0028877
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0029117, upper bound: 0.0029351
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0029243, upper bound: 0.0029227
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0018166, upper bound: 0.0018279
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0018166, upper bound: 0.0018279
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028431, upper bound: 0.0028511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0028431, upper bound: 0.0028511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0018174, upper bound: 0.0018255
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.09
Output dim: 4, lower bound: -0.0018174, upper bound: 0.0018255

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002365, 0.0002394
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013254, 0.0013092
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029250, 0.0029610
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012478, 0.0012326
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048409, 0.0047820
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009417, 0.0009303
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012106, 0.0012255
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001544, 0.0001563
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008467, 0.0008364
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041875, 0.0042390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027923, upper bound: 0.0027809
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027911, upper bound: 0.0027809
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002376, 0.0002383
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013194, 0.0013155
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029390, 0.0029477
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012422, 0.0012385
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048191, 0.0048049
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009375, 0.0009347
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012164, 0.0012200
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001552, 0.0001556
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008429, 0.0008405
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042075, 0.0042200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027954, upper bound: 0.0027728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027938, upper bound: 0.0027731
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002497, 0.0002514
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013920, 0.0013826
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030888, 0.0031100
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013106, 0.0013016
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050845, 0.0050498
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009891, 0.0009824
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012784, 0.0012872
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001631, 0.0001642
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008894, 0.0008833
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044220, 0.0044523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027786, upper bound: 0.0027945
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027802, upper bound: 0.0027821
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002499, 0.0002512
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013910, 0.0013837
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030913, 0.0031077
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013096, 0.0013027
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050806, 0.0050540
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009884, 0.0009832
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012795, 0.0012862
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001632, 0.0001641
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008887, 0.0008840
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044256, 0.0044490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027791, upper bound: 0.0027956
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027803, upper bound: 0.0027843
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002499, 0.0002510
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013900, 0.0013839
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030917, 0.0031054
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013086, 0.0013028
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050769, 0.0050545
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009877, 0.0009833
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012796, 0.0012853
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001632, 0.0001640
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008880, 0.0008841
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044261, 0.0044458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028132, upper bound: 0.0027595
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028155, upper bound: 0.0027558
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002501, 0.0002511
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013903, 0.0013846
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030932, 0.0031060
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013089, 0.0013035
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050780, 0.0050571
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009879, 0.0009838
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012803, 0.0012856
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001633, 0.0001640
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008882, 0.0008846
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044284, 0.0044467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027708
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028033, upper bound: 0.0027655
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002500, 0.0002508
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013886, 0.0013842
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030924, 0.0031023
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013073, 0.0013031
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050718, 0.0050557
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009867, 0.0009835
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012799, 0.0012840
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001633, 0.0001638
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008871, 0.0008843
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044271, 0.0044413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028119, upper bound: 0.0027597
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028152, upper bound: 0.0027559
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002503, 0.0002509
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013894, 0.0013857
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030959, 0.0031042
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013081, 0.0013046
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050749, 0.0050614
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009873, 0.0009847
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012814, 0.0012848
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001635, 0.0001639
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008877, 0.0008853
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044322, 0.0044440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027711
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028032, upper bound: 0.0027656
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002499, 0.0002513
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013917, 0.0013835
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030910, 0.0031092
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013102, 0.0013025
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050831, 0.0050534
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009889, 0.0009831
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012793, 0.0012869
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001632, 0.0001642
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008891, 0.0008839
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044251, 0.0044512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027913, upper bound: 0.0027809
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027954, upper bound: 0.0027737
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002497, 0.0002511
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013905, 0.0013828
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030892, 0.0031064
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013091, 0.0013018
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050786, 0.0050505
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009880, 0.0009825
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012786, 0.0012857
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001631, 0.0001640
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008883, 0.0008834
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044226, 0.0044472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027786, upper bound: 0.0027945
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027802, upper bound: 0.0027828
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002499, 0.0002511
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013901, 0.0013838
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030915, 0.0031057
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013088, 0.0013028
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050775, 0.0050542
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009878, 0.0009832
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012796, 0.0012854
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001632, 0.0001640
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008881, 0.0008841
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044259, 0.0044462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027909, upper bound: 0.0027809
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027938, upper bound: 0.0027742
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002500, 0.0002510
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013897, 0.0013841
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030922, 0.0031046
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013083, 0.0013030
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050757, 0.0050553
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009874, 0.0009835
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012798, 0.0012850
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001633, 0.0001639
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008878, 0.0008843
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044268, 0.0044447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027791, upper bound: 0.0027956
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027803, upper bound: 0.0027863
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002501, 0.0002509
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013894, 0.0013850
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030943, 0.0031040
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013080, 0.0013039
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050747, 0.0050588
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009872, 0.0009841
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012807, 0.0012847
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001634, 0.0001639
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008876, 0.0008849
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044299, 0.0044438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028132, upper bound: 0.0027595
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028155, upper bound: 0.0027563
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002503, 0.0002507
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013880, 0.0013857
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030958, 0.0031008
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013067, 0.0013046
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050695, 0.0050613
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009862, 0.0009846
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012813, 0.0012834
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001634, 0.0001637
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008867, 0.0008853
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044320, 0.0044392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028119, upper bound: 0.0027597
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028152, upper bound: 0.0027563
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002373, 0.0002387
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013217, 0.0013137
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029349, 0.0029528
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012443, 0.0012368
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048274, 0.0047982
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009391, 0.0009334
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012147, 0.0012221
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001550, 0.0001559
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008444, 0.0008393
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042017, 0.0042272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027708
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027711
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002383, 0.0002375
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013153, 0.0013194
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029476, 0.0029385
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012383, 0.0012421
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048041, 0.0048190
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009346, 0.0009375
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012200, 0.0012162
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001556, 0.0001551
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008403, 0.0008429
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042199, 0.0042068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028033, upper bound: 0.0027660
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0028032, upper bound: 0.0027667
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002506, 0.0002503
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013858, 0.0013874
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0030997, 0.0030961
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013047, 0.0013062
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050618, 0.0050676
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009847, 0.0009858
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012829, 0.0012815
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001637, 0.0001635
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008854, 0.0008864
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044375, 0.0044325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027863, upper bound: 0.0027803
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027956, upper bound: 0.0027791
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002508, 0.0002503
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013860, 0.0013886
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0031023, 0.0030965
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013049, 0.0013073
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050625, 0.0050719
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009848, 0.0009867
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012840, 0.0012816
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001638, 0.0001635
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008855, 0.0008871
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044413, 0.0044331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027742, upper bound: 0.0027938
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027809, upper bound: 0.0027909
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002508, 0.0002502
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013853, 0.0013889
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0031030, 0.0030949
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013042, 0.0013076
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050599, 0.0050731
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009843, 0.0009869
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012843, 0.0012810
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001638, 0.0001634
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008851, 0.0008874
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044424, 0.0044308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027843, upper bound: 0.0027803
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027956, upper bound: 0.0027791
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002509, 0.0002501
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013846, 0.0013891
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0031033, 0.0030934
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0013036, 0.0013077
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0050573, 0.0050735
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009838, 0.0009870
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012844, 0.0012803
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001638, 0.0001633
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008846, 0.0008874
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0044428, 0.0044285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027731, upper bound: 0.0027938
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027809, upper bound: 0.0027911
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002346, 0.0002357
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013052, 0.0012992
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029025, 0.0029160
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012288, 0.0012231
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047673, 0.0047452
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009274, 0.0009231
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012013, 0.0012069
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001532, 0.0001540
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008339, 0.0008300
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041553, 0.0041746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027660, upper bound: 0.0028033
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027655, upper bound: 0.0028033
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002348, 0.0002354
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013032, 0.0013003
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029050, 0.0029114
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012269, 0.0012242
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047598, 0.0047493
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009260, 0.0009239
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012024, 0.0012050
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001534, 0.0001537
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008326, 0.0008307
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041589, 0.0041680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027828, upper bound: 0.0027802
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027821, upper bound: 0.0027802
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002398, 0.0002386
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013211, 0.0013280
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029670, 0.0029514
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012437, 0.0012503
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048253, 0.0048507
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009387, 0.0009436
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012280, 0.0012216
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001566, 0.0001558
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008440, 0.0008485
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042476, 0.0042254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027708, upper bound: 0.0028024
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027945, upper bound: 0.0027786
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002401, 0.0002387
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0013218, 0.0013296
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0029705, 0.0029531
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012444, 0.0012518
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0048280, 0.0048565
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009392, 0.0009448
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0012295, 0.0012223
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001568, 0.0001559
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008445, 0.0008495
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0042527, 0.0042277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027708, upper bound: 0.0028024
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027945, upper bound: 0.0027786
time: 0.72 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 14.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027923, upper bound: 0.0027809
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027911, upper bound: 0.0027809
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027954, upper bound: 0.0027728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027938, upper bound: 0.0027731
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027786, upper bound: 0.0027945
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027802, upper bound: 0.0027821
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027791, upper bound: 0.0027956
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027803, upper bound: 0.0027843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028132, upper bound: 0.0027595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028155, upper bound: 0.0027558
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027708
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028033, upper bound: 0.0027655
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028119, upper bound: 0.0027597
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028152, upper bound: 0.0027559
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027711
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028032, upper bound: 0.0027656
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027913, upper bound: 0.0027809
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027954, upper bound: 0.0027737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027786, upper bound: 0.0027945
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027802, upper bound: 0.0027828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027909, upper bound: 0.0027809
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027938, upper bound: 0.0027742
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027791, upper bound: 0.0027956
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027803, upper bound: 0.0027863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028132, upper bound: 0.0027595
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028155, upper bound: 0.0027563
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028119, upper bound: 0.0027597
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028152, upper bound: 0.0027563
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027708
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028024, upper bound: 0.0027711
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028033, upper bound: 0.0027660
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0028032, upper bound: 0.0027667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027863, upper bound: 0.0027803
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027956, upper bound: 0.0027791
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027742, upper bound: 0.0027938
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027809, upper bound: 0.0027909
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027843, upper bound: 0.0027803
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027956, upper bound: 0.0027791
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027731, upper bound: 0.0027938
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027809, upper bound: 0.0027911
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027660, upper bound: 0.0028033
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027655, upper bound: 0.0028033
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027828, upper bound: 0.0027802
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027821, upper bound: 0.0027802
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027708, upper bound: 0.0028024
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027945, upper bound: 0.0027786
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027708, upper bound: 0.0028024
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.94
Output dim: 4, lower bound: -0.0027945, upper bound: 0.0027786

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002318, 0.0002347
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012995, 0.0012834
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028672, 0.0029033
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012235, 0.0012082
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047466, 0.0046875
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009234, 0.0009119
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011867, 0.0012017
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001514, 0.0001533
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008303, 0.0008199
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041047, 0.0041565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 150

Time for candidate selection: 11.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027909, upper bound: 0.0027136
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027451, upper bound: 0.0027795
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002318, 0.0002344
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012980, 0.0012834
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028673, 0.0028999
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012220, 0.0012083
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047409, 0.0046877
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009223, 0.0009119
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011868, 0.0012002
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001514, 0.0001531
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008293, 0.0008200
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041049, 0.0041515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 108

Time for candidate selection: 12.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027908, upper bound: 0.0027653
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027761, upper bound: 0.0027806
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002329, 0.0002336
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012936, 0.0012897
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028813, 0.0028900
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012179, 0.0012142
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047248, 0.0047106
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009192, 0.0009164
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011925, 0.0011962
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001521, 0.0001526
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008265, 0.0008240
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041249, 0.0041374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 231

Time for candidate selection: 11.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026365, upper bound: 0.0026796
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027028, upper bound: 0.0026325
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002329, 0.0002334
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012923, 0.0012897
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028813, 0.0028871
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012166, 0.0012142
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047201, 0.0047106
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009182, 0.0009164
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011926, 0.0011950
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001521, 0.0001524
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008256, 0.0008240
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041250, 0.0041333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 12.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027935, upper bound: 0.0027563
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027809, upper bound: 0.0027728
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002323, 0.0002347
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012993, 0.0012860
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028732, 0.0029028
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012232, 0.0012108
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047457, 0.0046973
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009232, 0.0009138
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011892, 0.0012014
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001517, 0.0001533
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008301, 0.0008216
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041133, 0.0041557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 188

Time for candidate selection: 12.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027783, upper bound: 0.0027788
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027640, upper bound: 0.0027942
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002329, 0.0002331
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012909, 0.0012898
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028816, 0.0028839
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012153, 0.0012143
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047149, 0.0047110
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009172, 0.0009165
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011927, 0.0011936
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001521, 0.0001523
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008247, 0.0008240
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041253, 0.0041287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 11.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027754, upper bound: 0.0027783
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027763, upper bound: 0.0027773
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002324, 0.0002345
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012983, 0.0012869
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028752, 0.0029004
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012223, 0.0012116
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047419, 0.0047006
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009225, 0.0009144
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011900, 0.0012005
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001518, 0.0001531
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008294, 0.0008222
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041162, 0.0041523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 150

Time for candidate selection: 11.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027742, upper bound: 0.0027917
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027753, upper bound: 0.0027910
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002332, 0.0002331
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012908, 0.0012909
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028841, 0.0028837
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012152, 0.0012154
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047145, 0.0047152
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009172, 0.0009173
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011937, 0.0011935
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001523, 0.0001522
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008246, 0.0008248
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041290, 0.0041283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 11.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027755, upper bound: 0.0027805
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0027764, upper bound: 0.0027794
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002320, 0.0002343
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012972, 0.0012849
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028705, 0.0028982
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012213, 0.0012096
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047382, 0.0046929
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009218, 0.0009130
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011881, 0.0011995
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001516, 0.0001530
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008288, 0.0008209
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041095, 0.0041491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 77

Time for candidate selection: 12.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 150

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018454, upper bound: 0.0018372
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0018454, upper bound: 0.0018372
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002332, 0.0002334
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012922, 0.0012911
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028845, 0.0028869
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012166, 0.0012155
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047198, 0.0047157
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009182, 0.0009174
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011939, 0.0011949
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001523, 0.0001524
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008256, 0.0008249
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041295, 0.0041330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 12.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0024094, upper bound: 0.0024046
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024594, upper bound: 0.0023572
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002326, 0.0002343
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012975, 0.0012877
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028768, 0.0028988
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012216, 0.0012123
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047392, 0.0047032
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009220, 0.0009150
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011907, 0.0011998
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001519, 0.0001530
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008290, 0.0008227
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041185, 0.0041500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 188
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 108
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 150
type: RSZ, layer: 3, pos: 168
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 11.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0024597, upper bound: 0.0026363
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0026689, upper bound: 0.0024685
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0040151, -0.0036103, -0.0040151, -0.0036103, -0.0002333, 0.0002329
1: -0.0002620, 0.0019796, -0.0002620, 0.0019796, -0.0012894, 0.0012918
2: 0.0105434, 0.0155515, 0.0105434, 0.0155515, -0.0028860, 0.0028808
3: 0.0007809, 0.0028913, 0.0007809, 0.0028913, -0.0012140, 0.0012162
4: 0.9997800, 1.0079676, 0.9997800, 1.0079676, -0.0047097, 0.0047183
5: 0.0021986, 0.0037914, 0.0021986, 0.0037914, -0.0009162, 0.0009179
6: -0.0106769, -0.0086041, -0.0106769, -0.0086041, -0.0011945, 0.0011923
7: -0.0101653, -0.0099009, -0.0101653, -0.0099009, -0.0001524, 0.0001521
8: -0.0049094, -0.0034772, -0.0049094, -0.0034772, -0.0008238, 0.0008253
9: -0.0007631, 0.0064065, -0.0007631, 0.0064065, -0.0041317, 0.0041242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.47 + 597.08 = 600.56 seconds
