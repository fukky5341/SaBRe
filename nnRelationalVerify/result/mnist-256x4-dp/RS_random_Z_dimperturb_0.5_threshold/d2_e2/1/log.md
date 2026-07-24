## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.35483528


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4698062, 0.4698062)
1: (-0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418)
2: (-0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225)
3: (-0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356)
4: (-0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974)
5: (-0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2713850, 0.2713850)
6: (-0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2700567, 0.2700567)
7: (-0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551)
8: (0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7120352, 0.7120352)
9: (-0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2857819, 0.2857819)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 3.11 = 4.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4251001, upper bound: 0.4251001

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3938211, upper bound: 0.3938211
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3938211, upper bound: 0.3938211
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 8, lower bound: -0.3938211, upper bound: 0.3938211
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 8, lower bound: -0.3938211, upper bound: 0.3938211

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4697280, 0.4696455
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2713193, 0.2712498
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2700088, 0.2699578
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7108126, 0.7114410
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2855705, 0.2853434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3902341, upper bound: 0.3921866
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3921866, upper bound: 0.3902341
time: 0.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4698062, 0.4697280
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2713850, 0.2713193
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2700567, 0.2700088
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7114410, 0.7120352
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2857819, 0.2855705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3902341, upper bound: 0.3921866
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3921866, upper bound: 0.3902341
time: 0.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.3902341, upper bound: 0.3921866
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.3921866, upper bound: 0.3902341
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.3902341, upper bound: 0.3921866
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 8, lower bound: -0.3921866, upper bound: 0.3902341

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693145, 0.4690430
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709394, 0.2707112
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697340, 0.2695668
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7058253, 0.7078876
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2844774, 0.2837324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3789584, upper bound: 0.3866534
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3846918, upper bound: 0.3810450
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691259, 0.4691868
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707807, 0.2708320
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696178, 0.2696553
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7069173, 0.7064543
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2839595, 0.2841268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3810450, upper bound: 0.3846918
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3866534, upper bound: 0.3789584
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693931, 0.4691257
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2710055, 0.2707807
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697820, 0.2696178
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7064543, 0.7084961
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2846931, 0.2839595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3789584, upper bound: 0.3866534
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3846918, upper bound: 0.3810450
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692045, 0.4693143
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708469, 0.2709395
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696658, 0.2697340
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7078876, 0.7070627
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2841754, 0.2844774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3810450, upper bound: 0.3846918
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3866534, upper bound: 0.3789584
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3789584, upper bound: 0.3866534
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3846918, upper bound: 0.3810450
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3810450, upper bound: 0.3846918
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3866534, upper bound: 0.3789584
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3789584, upper bound: 0.3866534
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3846918, upper bound: 0.3810450
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3810450, upper bound: 0.3846918
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.60
Output dim: 8, lower bound: -0.3866534, upper bound: 0.3789584

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693079, 0.4690354
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709342, 0.2707050
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697307, 0.2695626
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7057686, 0.7078404
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2844591, 0.2837108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 3.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3768977, upper bound: 0.3046070
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3079376, upper bound: 0.3849126
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693067, 0.4690359
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709333, 0.2707055
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697300, 0.2695630
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7057734, 0.7078314
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2844557, 0.2837124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 3.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3843197, upper bound: 0.3543033
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3396714, upper bound: 0.3805909
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691188, 0.4691789
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707752, 0.2708258
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696141, 0.2696512
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7068605, 0.7064037
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2839402, 0.2841052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3666625, upper bound: 0.3718061
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3671901, upper bound: 0.3677526
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691181, 0.4691799
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707746, 0.2708266
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696136, 0.2696518
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7068682, 0.7063975
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2839379, 0.2841078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 3.92 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3795274, upper bound: 0.3715393
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3787100, upper bound: 0.3718099
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693866, 0.4691181
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2710003, 0.2707745
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697785, 0.2696136
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7063980, 0.7084498
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2846750, 0.2839379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 3.90 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3781114, upper bound: 0.3778242
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3675326, upper bound: 0.3859108
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693854, 0.4691188
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709993, 0.2707753
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697778, 0.2696141
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7064037, 0.7084408
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2846716, 0.2839402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 3.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3807528, upper bound: 0.3776226
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3817118, upper bound: 0.3779175
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691975, 0.4693067
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708412, 0.2709333
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696619, 0.2697300
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7078314, 0.7070131
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2841561, 0.2844558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 3.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3665246, upper bound: 0.3633304
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3577744, upper bound: 0.3706071
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691968, 0.4693077
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708405, 0.2709342
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696614, 0.2697307
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7078404, 0.7070069
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2841538, 0.2844590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3862715, upper bound: 0.3397908
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3544772, upper bound: 0.3785262
time: 0.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3768977, upper bound: 0.3046070
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3079376, upper bound: 0.3849126
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3843197, upper bound: 0.3543033
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3396714, upper bound: 0.3805909
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3666625, upper bound: 0.3718061
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3671901, upper bound: 0.3677526
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3795274, upper bound: 0.3715393
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3787100, upper bound: 0.3718099
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3781114, upper bound: 0.3778242
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3675326, upper bound: 0.3859108
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3807528, upper bound: 0.3776226
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3817118, upper bound: 0.3779175
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3665246, upper bound: 0.3633304
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3577744, upper bound: 0.3706071
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3862715, upper bound: 0.3397908
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.61
Output dim: 8, lower bound: -0.3544772, upper bound: 0.3785262

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684860, 0.4684834
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702173, 0.2702154
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692037, 0.2692021
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7012367, 0.7012544
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2827121, 0.2827059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3717178, upper bound: 0.2968310
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3651870, upper bound: 0.2979153
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687949, 0.4682136
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2704774, 0.2699880
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693940, 0.2690356
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6991830, 0.7036037
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2835609, 0.2819639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2935870, upper bound: 0.3715910
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2948136, upper bound: 0.3687309
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689999, 0.4688251
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706738, 0.2705266
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695457, 0.2694378
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7042089, 0.7055383
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836192, 0.2831392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3825286, upper bound: 0.3031111
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3065185, upper bound: 0.3520262
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690821, 0.4687293
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707428, 0.2704461
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695962, 0.2693788
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7034802, 0.7061625
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2838448, 0.2828758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2999247, upper bound: 0.3043613
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2943580, upper bound: 0.3145911
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682356, 0.4682760
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700477, 0.2700818
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691244, 0.2691493
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7004213, 0.7001140
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2814564, 0.2815678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3515998, upper bound: 0.3418004
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3370420, upper bound: 0.3562733
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682158, 0.4682500
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700312, 0.2700598
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691123, 0.2691332
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7002234, 0.6999643
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2814029, 0.2814958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2924474, upper bound: 0.2858330
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2853400, upper bound: 0.2925085
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689364, 0.4690938
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706492, 0.2707816
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695493, 0.2696464
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7067351, 0.7055402
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832284, 0.2836597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3633528, upper bound: 0.3582619
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3664896, upper bound: 0.3562238
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690320, 0.4689782
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707295, 0.2706842
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696083, 0.2695749
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7058563, 0.7062650
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834898, 0.2833425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3658937, upper bound: 0.3437516
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3603080, upper bound: 0.3581209
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692746, 0.4690914
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708849, 0.2707312
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696746, 0.2695625
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7058554, 0.7072563
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2845478, 0.2840470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3776781, upper bound: 0.3411660
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3389198, upper bound: 0.3773984
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693626, 0.4690061
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709588, 0.2706590
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697287, 0.2695094
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7052040, 0.7079253
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2847894, 0.2838112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3653269, upper bound: 0.3051742
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3046763, upper bound: 0.3841764
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689143, 0.4687934
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707291, 0.2706258
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696529, 0.2695789
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055039, 0.7064362
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2835602, 0.2832267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3664605, upper bound: 0.3549072
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3517794, upper bound: 0.3636756
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690595, 0.4687204
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708504, 0.2705643
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697423, 0.2695336
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7049503, 0.7075338
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2839587, 0.2830248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3813385, upper bound: 0.3509356
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3254032, upper bound: 0.3774635
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682412, 0.4687984
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701294, 0.2705988
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691723, 0.2695168
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7050838, 0.7008560
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2814525, 0.2829829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3658598, upper bound: 0.3615360
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3565680, upper bound: 0.3624526
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686894, 0.4683952
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705068, 0.2702591
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694488, 0.2692682
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7020173, 0.7042654
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2826834, 0.2818754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3529176, upper bound: 0.3679396
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3549072, upper bound: 0.3664605
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688904, 0.4690893
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705812, 0.2707490
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694772, 0.2696006
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7062173, 0.7047129
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2833179, 0.2838647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3830860, upper bound: 0.3251830
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3832532, upper bound: 0.3364781
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689901, 0.4690011
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706651, 0.2706748
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695387, 0.2695464
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055473, 0.7054720
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2835921, 0.2836224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3431590, upper bound: 0.3507812
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3280231, upper bound: 0.3640668
time: 0.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3717178, upper bound: 0.2968310
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3651870, upper bound: 0.2979153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.2935870, upper bound: 0.3715910
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.2948136, upper bound: 0.3687309
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3825286, upper bound: 0.3031111
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3065185, upper bound: 0.3520262
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.2999247, upper bound: 0.3043613
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.2943580, upper bound: 0.3145911
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3515998, upper bound: 0.3418004
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3370420, upper bound: 0.3562733
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.2924474, upper bound: 0.2858330
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.2853400, upper bound: 0.2925085
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3633528, upper bound: 0.3582619
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3664896, upper bound: 0.3562238
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3658937, upper bound: 0.3437516
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3603080, upper bound: 0.3581209
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3776781, upper bound: 0.3411660
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3389198, upper bound: 0.3773984
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3653269, upper bound: 0.3051742
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3046763, upper bound: 0.3841764
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3664605, upper bound: 0.3549072
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3517794, upper bound: 0.3636756
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3813385, upper bound: 0.3509356
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3254032, upper bound: 0.3774635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3658598, upper bound: 0.3615360
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3565680, upper bound: 0.3624526
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3529176, upper bound: 0.3679396
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3549072, upper bound: 0.3664605
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3830860, upper bound: 0.3251830
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3832532, upper bound: 0.3364781
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3431590, upper bound: 0.3507812
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 8, lower bound: -0.3280231, upper bound: 0.3640668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686418, 0.4685330
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703480, 0.2702569
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693011, 0.2692342
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7015996, 0.7024236
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2830464, 0.2827492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3538475, upper bound: 0.2949297
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3701933, upper bound: 0.2939672
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688456, 0.4683692
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705197, 0.2701187
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694268, 0.2691331
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7003522, 0.7039742
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836068, 0.2822982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543900, upper bound: 0.2883747
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3425892, upper bound: 0.2892583
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684001, 0.4681325
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701861, 0.2699609
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692257, 0.2690609
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6993294, 0.7013638
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2819078, 0.2811735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2864490, upper bound: 0.3560265
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2860322, upper bound: 0.3667892
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684049, 0.4681308
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701903, 0.2699594
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692288, 0.2690598
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6993175, 0.7014010
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2819217, 0.2811686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2942193, upper bound: 0.3378823
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2908548, upper bound: 0.3683328
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684848, 0.4684925
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702163, 0.2702230
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692028, 0.2692076
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7013059, 0.7012458
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2827089, 0.2827307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3653648, upper bound: 0.2895341
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3696482, upper bound: 0.2888064
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685265, 0.4682574
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702166, 0.2699901
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691833, 0.2690173
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6989460, 0.7009919
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2828432, 0.2821041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2881877, upper bound: 0.2793827
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2809573, upper bound: 0.2895097
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4681751, 0.4682770
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2699967, 0.2700826
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2690870, 0.2691499
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7004280, 0.6996529
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2812898, 0.2815704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3615928, upper bound: 0.2866480
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2831633, upper bound: 0.3562533
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682151, 0.4683049
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700305, 0.2701061
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691118, 0.2691672
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7006412, 0.6999581
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2814006, 0.2816468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3534740, upper bound: 0.3339632
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3468781, upper bound: 0.3428260
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4681575, 0.4686718
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700589, 0.2704921
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691213, 0.2694386
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7041202, 0.7002089
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2812221, 0.2826350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3641279, upper bound: 0.2875008
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2910392, upper bound: 0.3416120
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686100, 0.4682755
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2704400, 0.2701584
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694005, 0.2691942
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7011075, 0.7036498
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2824652, 0.2815466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3506154, upper bound: 0.3553937
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3576568, upper bound: 0.3553235
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690802, 0.4689097
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707409, 0.2705979
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695943, 0.2694899
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7048521, 0.7061558
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2838391, 0.2833717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3705596, upper bound: 0.3169148
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3702859, upper bound: 0.3341597
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691696, 0.4688113
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708161, 0.2705151
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696493, 0.2694294
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7041049, 0.7068372
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2840852, 0.2831013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3315049, upper bound: 0.3706756
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3127196, upper bound: 0.3691223
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685649, 0.4685769
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702838, 0.2702941
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692515, 0.2692598
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7019477, 0.7018628
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2829304, 0.2829627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3638998, upper bound: 0.3036274
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3644075, upper bound: 0.3041242
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688658, 0.4682961
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705365, 0.2700576
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694366, 0.2690866
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6998119, 0.7041459
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837548, 0.2821910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2942785, upper bound: 0.3639280
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2954218, upper bound: 0.3703872
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684641, 0.4686108
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703170, 0.2704408
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693097, 0.2694011
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7036562, 0.7025511
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2820647, 0.2824674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3622585, upper bound: 0.3476720
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3493742, upper bound: 0.3493228
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688773, 0.4681723
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706648, 0.2700714
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695646, 0.2691306
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7003226, 0.7056930
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2831990, 0.2812632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 255

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3466302, upper bound: 0.3520380
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3387877, upper bound: 0.3591407
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690790, 0.4689147
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707400, 0.2706020
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695935, 0.2694930
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7048898, 0.7061467
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2838358, 0.2833853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3599051, upper bound: 0.3491729
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3798859, upper bound: 0.3481162
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691622, 0.4688122
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708101, 0.2705158
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696449, 0.2694299
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7041106, 0.7067819
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2840652, 0.2831036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3186267, upper bound: 0.3673598
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3111908, upper bound: 0.3702072
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690858, 0.4692750
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707258, 0.2708857
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695580, 0.2696757
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7072515, 0.7058196
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2840290, 0.2845511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3520001, upper bound: 0.3600627
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3643857, upper bound: 0.3403925
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691771, 0.4691947
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708029, 0.2708177
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696144, 0.2696257
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7066379, 0.7065167
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2842806, 0.2843291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3471628, upper bound: 0.3387379
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3389824, upper bound: 0.3519442
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688017, 0.4689810
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706344, 0.2707838
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695836, 0.2696948
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7069311, 0.7055807
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832512, 0.2837422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3520334, upper bound: 0.3612905
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3407299, upper bound: 0.3672523
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688716, 0.4688330
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706923, 0.2706590
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696266, 0.2696031
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7058058, 0.7061062
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834431, 0.2833338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3544477, upper bound: 0.3220164
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3207035, upper bound: 0.3660408
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687667, 0.4689822
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706050, 0.2707848
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695620, 0.2696955
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7069407, 0.7053151
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2831552, 0.2837455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3658582, upper bound: 0.3234760
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3816060, upper bound: 0.3209450
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688709, 0.4688668
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706918, 0.2706877
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696260, 0.2696241
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7060642, 0.7061000
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834409, 0.2834271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3671077, upper bound: 0.3228254
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700219, upper bound: 0.3221886
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685782, 0.4683862
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702605, 0.2700986
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692148, 0.2690969
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6999254, 0.7013953
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2829884, 0.2824579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3207650, upper bound: 0.3580050
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3089969, upper bound: 0.3580262
time: 0.80 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3538475, upper bound: 0.2949297
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3701933, upper bound: 0.2939672
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3543900, upper bound: 0.2883747
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3425892, upper bound: 0.2892583
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2864490, upper bound: 0.3560265
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2860322, upper bound: 0.3667892
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2942193, upper bound: 0.3378823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2908548, upper bound: 0.3683328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3653648, upper bound: 0.2895341
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3696482, upper bound: 0.2888064
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2881877, upper bound: 0.2793827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2809573, upper bound: 0.2895097
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3615928, upper bound: 0.2866480
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2831633, upper bound: 0.3562533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3534740, upper bound: 0.3339632
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3468781, upper bound: 0.3428260
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3641279, upper bound: 0.2875008
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2910392, upper bound: 0.3416120
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3506154, upper bound: 0.3553937
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3576568, upper bound: 0.3553235
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3705596, upper bound: 0.3169148
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3702859, upper bound: 0.3341597
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3315049, upper bound: 0.3706756
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3127196, upper bound: 0.3691223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3638998, upper bound: 0.3036274
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3644075, upper bound: 0.3041242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2942785, upper bound: 0.3639280
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.2954218, upper bound: 0.3703872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3622585, upper bound: 0.3476720
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3493742, upper bound: 0.3493228
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3466302, upper bound: 0.3520380
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3387877, upper bound: 0.3591407
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3599051, upper bound: 0.3491729
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3798859, upper bound: 0.3481162
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3186267, upper bound: 0.3673598
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3111908, upper bound: 0.3702072
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3520001, upper bound: 0.3600627
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3643857, upper bound: 0.3403925
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3471628, upper bound: 0.3387379
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3389824, upper bound: 0.3519442
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3520334, upper bound: 0.3612905
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3407299, upper bound: 0.3672523
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3544477, upper bound: 0.3220164
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3207035, upper bound: 0.3660408
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3658582, upper bound: 0.3234760
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3816060, upper bound: 0.3209450
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3671077, upper bound: 0.3228254
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3700219, upper bound: 0.3221886
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3207650, upper bound: 0.3580050
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 8, lower bound: -0.3089969, upper bound: 0.3580262

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684516, 0.4684463
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702500, 0.2702452
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692266, 0.2692231
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7014604, 0.7015023
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2826525, 0.2826371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3693458, upper bound: 0.2928138
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3578851, upper bound: 0.2919266
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686418, 0.4685330
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703480, 0.2702569
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693011, 0.2692342
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7015996, 0.7024236
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2830464, 0.2827492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2768052, upper bound: 0.3406928
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2771629, upper bound: 0.3434821
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688456, 0.4683692
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705197, 0.2701187
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694268, 0.2691331
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7003522, 0.7039742
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836068, 0.2822982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2810313, upper bound: 0.3639908
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2832954, upper bound: 0.3637244
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690893, 0.4687285
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707489, 0.2704455
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696007, 0.2693784
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7034760, 0.7062173
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2838647, 0.2828742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2668088, upper bound: 0.2837650
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2668056, upper bound: 0.2913594
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4683474, 0.4681330
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701418, 0.2699614
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691933, 0.2690612
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6993337, 0.7009633
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2817631, 0.2811751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3645641, upper bound: 0.2881222
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3645433, upper bound: 0.2885251
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684037, 0.4681828
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701892, 0.2700032
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692281, 0.2690917
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6997123, 0.7013919
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2819184, 0.2813112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3494577, upper bound: 0.2868628
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3681869, upper bound: 0.2868399
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682962, 0.4686589
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700576, 0.2703632
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2690866, 0.2693104
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7025719, 0.6998119
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2821910, 0.2831881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3472565, upper bound: 0.2772550
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3288788, upper bound: 0.2776406
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685770, 0.4683580
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702940, 0.2701097
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692598, 0.2691247
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7002821, 0.7019477
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2829627, 0.2823609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2769330, upper bound: 0.3439841
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2745828, upper bound: 0.3511876
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682962, 0.4686589
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700576, 0.2703632
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2690866, 0.2693104
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7025719, 0.6998119
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2821910, 0.2831881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3637205, upper bound: 0.2857361
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3050386, upper bound: 0.2868731
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686854, 0.4688544
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705349, 0.2706772
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695121, 0.2696166
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7059679, 0.7046843
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2829287, 0.2833943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3498369, upper bound: 0.3390563
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3397451, upper bound: 0.3547088
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687924, 0.4687653
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706251, 0.2706020
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695784, 0.2695613
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7052898, 0.7054977
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832245, 0.2831475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3403562, upper bound: 0.3538523
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3562067, upper bound: 0.3387090
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691806, 0.4690320
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708541, 0.2707295
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696988, 0.2696083
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7062645, 0.7074060
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2838955, 0.2834898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3654463, upper bound: 0.2994957
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3589422, upper bound: 0.3116467
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693007, 0.4689364
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709551, 0.2706492
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697729, 0.2695493
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055407, 0.7083168
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2842249, 0.2832284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3682798, upper bound: 0.2955398
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2972608, upper bound: 0.3317857
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691806, 0.4690320
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708541, 0.2707295
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696988, 0.2696083
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7062645, 0.7074060
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2838955, 0.2834898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3176227, upper bound: 0.3569559
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2977268, upper bound: 0.3587192
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4693007, 0.4689364
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709551, 0.2706492
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697729, 0.2695493
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055407, 0.7083168
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2842249, 0.2832284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3100986, upper bound: 0.2956675
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2950041, upper bound: 0.3671210
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692570, 0.4690039
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709046, 0.2706919
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697124, 0.2695572
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056465, 0.7075782
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843310, 0.2836310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3579461, upper bound: 0.2957143
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3557784, upper bound: 0.2966860
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692796, 0.4689884
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709238, 0.2706789
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697263, 0.2695475
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055292, 0.7077494
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843930, 0.2835884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3454008, upper bound: 0.2906404
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3523108, upper bound: 0.2898239
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684818, 0.4686100
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703319, 0.2704400
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693207, 0.2694005
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7036495, 0.7026861
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2821134, 0.2824651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2800131, upper bound: 0.3509126
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2810652, upper bound: 0.3486738
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688783, 0.4681573
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706657, 0.2700589
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695653, 0.2691214
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7002091, 0.7057021
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832022, 0.2812221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2943548, upper bound: 0.3691954
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2936337, upper bound: 0.3696604
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687192, 0.4686160
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2704136, 0.2703267
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693484, 0.2692854
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7022309, 0.7030239
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832605, 0.2829771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3614839, upper bound: 0.3459152
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3612656, upper bound: 0.3467958
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689057, 0.4684527
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705709, 0.2701890
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694635, 0.2691846
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7009869, 0.7044401
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837726, 0.2825276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3275292, upper bound: 0.3576277
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3371644, upper bound: 0.3410497
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687911, 0.4682627
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705348, 0.2700909
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694349, 0.2691101
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7000661, 0.7040911
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2835823, 0.2821336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3590227, upper bound: 0.3353597
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3495938, upper bound: 0.3483459
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685296, 0.4685426
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703149, 0.2703261
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692738, 0.2692823
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7021914, 0.7021022
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2828643, 0.2829012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3780845, upper bound: 0.2997113
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2995146, upper bound: 0.3458379
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691739, 0.4690328
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708484, 0.2707303
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696947, 0.2696089
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7062712, 0.7073550
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2838770, 0.2834921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3171364, upper bound: 0.3662349
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3176208, upper bound: 0.3665576
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692996, 0.4689415
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709541, 0.2706533
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697721, 0.2695522
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055769, 0.7083077
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2842216, 0.2832416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3101026, upper bound: 0.3691305
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3101969, upper bound: 0.3694121
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686290, 0.4684505
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703981, 0.2702489
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693348, 0.2692258
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7014937, 0.7028575
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2831368, 0.2826492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3369751, upper bound: 0.3478678
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3391839, upper bound: 0.3401959
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4683419, 0.4687047
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701570, 0.2704626
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691581, 0.2693825
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7034254, 0.7006750
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2823488, 0.2833468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3578483, upper bound: 0.3342901
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3565626, upper bound: 0.3248213
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690858, 0.4692750
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707258, 0.2708857
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695580, 0.2696757
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7072515, 0.7058196
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2840290, 0.2845511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3515515, upper bound: 0.3113961
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3236209, upper bound: 0.3608645
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691771, 0.4691947
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708029, 0.2708177
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696144, 0.2696257
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7066379, 0.7065167
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2842806, 0.2843291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3401877, upper bound: 0.3121247
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3214165, upper bound: 0.3668499
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689949, 0.4689999
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706693, 0.2706738
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695417, 0.2695457
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055383, 0.7055097
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836057, 0.2836192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3129404, upper bound: 0.3278276
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3073729, upper bound: 0.3533350
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686149, 0.4684515
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703865, 0.2702499
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693262, 0.2692266
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7015023, 0.7027516
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2830986, 0.2826524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3609535, upper bound: 0.3117811
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3472450, upper bound: 0.3175172
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4683410, 0.4687209
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701563, 0.2704762
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691576, 0.2693924
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7035475, 0.7006688
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2823465, 0.2833909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3678616, upper bound: 0.2981188
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3580032, upper bound: 0.3085304
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682521, 0.4684048
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700614, 0.2701902
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691326, 0.2692288
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7014008, 0.7002409
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2815008, 0.2819216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3600086, upper bound: 0.3001394
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3576303, upper bound: 0.3157272
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682938, 0.4684000
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700969, 0.2701861
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691585, 0.2692257
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7013640, 0.7005613
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2816165, 0.2819078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3653810, upper bound: 0.3010021
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3547888, upper bound: 0.3172191
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690280, 0.4692216
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707256, 0.2708892
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696047, 0.2697253
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7077074, 0.7062459
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834764, 0.2840109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3064770, upper bound: 0.3447202
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3068648, upper bound: 0.3427104
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691110, 0.4690890
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707955, 0.2707776
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696559, 0.2696433
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7067003, 0.7068739
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837038, 0.2836474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3054010, upper bound: 0.3555010
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3063024, upper bound: 0.3552105
time: 0.87 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 8.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3693458, upper bound: 0.2928138
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3578851, upper bound: 0.2919266
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2768052, upper bound: 0.3406928
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2771629, upper bound: 0.3434821
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2810313, upper bound: 0.3639908
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2832954, upper bound: 0.3637244
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2668088, upper bound: 0.2837650
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2668056, upper bound: 0.2913594
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3645641, upper bound: 0.2881222
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3645433, upper bound: 0.2885251
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3494577, upper bound: 0.2868628
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3681869, upper bound: 0.2868399
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3472565, upper bound: 0.2772550
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3288788, upper bound: 0.2776406
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2769330, upper bound: 0.3439841
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2745828, upper bound: 0.3511876
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3637205, upper bound: 0.2857361
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3050386, upper bound: 0.2868731
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3498369, upper bound: 0.3390563
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3397451, upper bound: 0.3547088
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3403562, upper bound: 0.3538523
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3562067, upper bound: 0.3387090
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3654463, upper bound: 0.2994957
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3589422, upper bound: 0.3116467
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3682798, upper bound: 0.2955398
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2972608, upper bound: 0.3317857
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3176227, upper bound: 0.3569559
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2977268, upper bound: 0.3587192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3100986, upper bound: 0.2956675
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2950041, upper bound: 0.3671210
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3579461, upper bound: 0.2957143
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3557784, upper bound: 0.2966860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3454008, upper bound: 0.2906404
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3523108, upper bound: 0.2898239
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2800131, upper bound: 0.3509126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2810652, upper bound: 0.3486738
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2943548, upper bound: 0.3691954
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2936337, upper bound: 0.3696604
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3614839, upper bound: 0.3459152
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3612656, upper bound: 0.3467958
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3275292, upper bound: 0.3576277
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3371644, upper bound: 0.3410497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3590227, upper bound: 0.3353597
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3495938, upper bound: 0.3483459
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3780845, upper bound: 0.2997113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.2995146, upper bound: 0.3458379
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3171364, upper bound: 0.3662349
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3176208, upper bound: 0.3665576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3101026, upper bound: 0.3691305
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3101969, upper bound: 0.3694121
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3369751, upper bound: 0.3478678
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3391839, upper bound: 0.3401959
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3578483, upper bound: 0.3342901
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3565626, upper bound: 0.3248213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3515515, upper bound: 0.3113961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3236209, upper bound: 0.3608645
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3401877, upper bound: 0.3121247
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3214165, upper bound: 0.3668499
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3129404, upper bound: 0.3278276
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3073729, upper bound: 0.3533350
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3609535, upper bound: 0.3117811
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3472450, upper bound: 0.3175172
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3678616, upper bound: 0.2981188
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3580032, upper bound: 0.3085304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3600086, upper bound: 0.3001394
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3576303, upper bound: 0.3157272
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3653810, upper bound: 0.3010021
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3547888, upper bound: 0.3172191
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3064770, upper bound: 0.3447202
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3068648, upper bound: 0.3427104
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3054010, upper bound: 0.3555010
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.82
Output dim: 8, lower bound: -0.3063024, upper bound: 0.3552105

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691957, 0.4690058
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708187, 0.2706591
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696265, 0.2695097
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7052040, 0.7066469
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843324, 0.2838118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3567588, upper bound: 0.2832159
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3399441, upper bound: 0.2838770
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692804, 0.4689233
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708901, 0.2705894
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696790, 0.2694585
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7045755, 0.7072906
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2845655, 0.2835841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3385485, upper bound: 0.2768956
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3458161, upper bound: 0.2769105
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688671, 0.4687097
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706877, 0.2705555
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696241, 0.2695274
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7048693, 0.7060642
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834271, 0.2829974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2736200, upper bound: 0.3567433
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2747641, upper bound: 0.3574542
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689822, 0.4686201
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707847, 0.2704799
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696955, 0.2694718
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7041874, 0.7069407
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837456, 0.2827492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2735499, upper bound: 0.3395827
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2742378, upper bound: 0.3508419
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691771, 0.4689195
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708375, 0.2706208
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696638, 0.2695051
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7050037, 0.7069631
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2841063, 0.2833990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3575048, upper bound: 0.2818219
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3568409, upper bound: 0.2810132
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691998, 0.4689062
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708566, 0.2706097
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696778, 0.2694968
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7049050, 0.7071338
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2841683, 0.2833629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3637466, upper bound: 0.2874453
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3496709, upper bound: 0.2872339
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684504, 0.4684591
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702489, 0.2702559
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692258, 0.2692310
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7015572, 0.7014937
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2826492, 0.2826720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543320, upper bound: 0.2771570
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3468271, upper bound: 0.2770626
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688113, 0.4689622
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705151, 0.2706420
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694294, 0.2695222
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7052507, 0.7041044
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2831013, 0.2835156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3430530, upper bound: 0.2838342
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3623051, upper bound: 0.2838145
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682618, 0.4685874
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700902, 0.2703637
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691096, 0.2693100
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7025318, 0.7000599
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2821313, 0.2830241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 255

### Candidate
type: RSZ, layer: 3, pos: 230

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3557572, upper bound: 0.3129398
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3027823, upper bound: 0.3381835
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687204, 0.4686093
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2704147, 0.2703209
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693492, 0.2692811
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7021775, 0.7030330
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832637, 0.2829580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3534008, upper bound: 0.2894772
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3355848, upper bound: 0.2872739
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689152, 0.4684520
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705790, 0.2701883
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694695, 0.2691841
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7009807, 0.7045131
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837991, 0.2825253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3531364, upper bound: 0.3089796
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3560222, upper bound: 0.3088822
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685649, 0.4685769
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702838, 0.2702941
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692515, 0.2692598
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7019477, 0.7018628
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2829304, 0.2829627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3674607, upper bound: 0.2940390
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3674003, upper bound: 0.2945333
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4684818, 0.4686100
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703319, 0.2704400
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693207, 0.2694005
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7036495, 0.7026861
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2821134, 0.2824651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3151512, upper bound: 0.3555057
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3158696, upper bound: 0.3387697
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688783, 0.4681573
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706657, 0.2700589
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695653, 0.2691214
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7002091, 0.7057021
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832022, 0.2812221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2922330, upper bound: 0.3477708
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2853519, upper bound: 0.3536985
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688658, 0.4682961
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705365, 0.2700576
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694366, 0.2690866
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6998119, 0.7041459
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837548, 0.2821910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2868483, upper bound: 0.3419224
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2856043, upper bound: 0.3565154
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687204, 0.4686093
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2704147, 0.2703209
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693492, 0.2692811
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7021775, 0.7030330
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2832637, 0.2829580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3072620, upper bound: 0.2770984
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2957362, upper bound: 0.2770984
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689152, 0.4684520
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705790, 0.2701883
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694695, 0.2691841
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7009807, 0.7045131
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837991, 0.2825253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3400644, upper bound: 0.2907136
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3493125, upper bound: 0.2884243
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692570, 0.4690039
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709046, 0.2706919
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697124, 0.2695572
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056465, 0.7075782
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843310, 0.2836310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 230

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2857420, upper bound: 0.3387089
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2852545, upper bound: 0.3571104
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692796, 0.4689884
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709238, 0.2706789
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697263, 0.2695475
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055292, 0.7077494
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843930, 0.2835884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2722499, upper bound: 0.2892733
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2722499, upper bound: 0.3024423
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692556, 0.4690065
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709035, 0.2706942
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697117, 0.2695589
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056675, 0.7075691
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843278, 0.2836385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 230

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3496527, upper bound: 0.3289376
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3196271, upper bound: 0.3374940
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692756, 0.4689891
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709202, 0.2706796
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697238, 0.2695479
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055354, 0.7077174
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843814, 0.2835907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3270979, upper bound: 0.3452662
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3598361, upper bound: 0.3307681
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687911, 0.4682627
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705348, 0.2700909
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694349, 0.2691101
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7000661, 0.7040911
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2835823, 0.2821336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3132045, upper bound: 0.3441110
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3133884, upper bound: 0.3429460
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692734, 0.4690950
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708838, 0.2707342
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696737, 0.2695646
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7058821, 0.7072473
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2845446, 0.2840567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3452414, upper bound: 0.3214390
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3233361, upper bound: 0.3272876
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685637, 0.4685869
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702828, 0.2703024
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692508, 0.2692658
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7020230, 0.7018538
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2829270, 0.2829897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3643565, upper bound: 0.2899726
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3567080, upper bound: 0.2900102
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692556, 0.4690065
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709035, 0.2706942
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697117, 0.2695589
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056675, 0.7075691
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843278, 0.2836385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2853496, upper bound: 0.2916595
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2818134, upper bound: 0.3031622
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692756, 0.4689891
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709202, 0.2706796
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697238, 0.2695479
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055354, 0.7077174
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843814, 0.2835907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3166681, upper bound: 0.3477306
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3157185, upper bound: 0.3657510
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692556, 0.4690065
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709035, 0.2706942
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697117, 0.2695589
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056675, 0.7075691
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843278, 0.2836385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3074021, upper bound: 0.2942200
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2911563, upper bound: 0.3671405
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692756, 0.4689891
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709202, 0.2706796
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697238, 0.2695479
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055354, 0.7077174
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843814, 0.2835907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2921368, upper bound: 0.3557593
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2991503, upper bound: 0.3550005
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690328, 0.4692206
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707298, 0.2708882
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696077, 0.2697246
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7076988, 0.7062826
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834897, 0.2840075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3433116, upper bound: 0.3230461
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3443787, upper bound: 0.3060311
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691117, 0.4690824
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707962, 0.2707720
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696564, 0.2696393
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7066498, 0.7068806
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837061, 0.2836289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2961168, upper bound: 0.2683930
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2862082, upper bound: 0.2789445
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689949, 0.4689999
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706693, 0.2706738
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695417, 0.2695457
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055383, 0.7055097
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836057, 0.2836192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3091794, upper bound: 0.3488263
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3093525, upper bound: 0.3385325
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4689949, 0.4689999
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706693, 0.2706738
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695417, 0.2695457
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055383, 0.7055097
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836057, 0.2836192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3160349, upper bound: 0.3545673
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3059343, upper bound: 0.3623549
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685307, 0.4688454
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702550, 0.2705197
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692321, 0.2694268
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7039742, 0.7015901
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2827426, 0.2836068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3602301, upper bound: 0.3106432
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3444597, upper bound: 0.3088049
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682262, 0.4687998
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2701167, 0.2705997
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691631, 0.2695175
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7050924, 0.7007425
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2814114, 0.2829863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3661268, upper bound: 0.2875134
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2882173, upper bound: 0.2954512
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4686887, 0.4684129
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705061, 0.2702740
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694483, 0.2692790
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7021527, 0.7042592
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2826811, 0.2819241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 248

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3572397, upper bound: 0.3065165
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3516968, upper bound: 0.3075627
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690280, 0.4692216
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707256, 0.2708892
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696047, 0.2697253
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7077074, 0.7062459
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834764, 0.2840109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3470031, upper bound: 0.2820907
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3414256, upper bound: 0.2892342
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691110, 0.4690890
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707955, 0.2707776
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696559, 0.2696433
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7067003, 0.7068739
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2837038, 0.2836474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556539, upper bound: 0.2812126
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2810871, upper bound: 0.3134068
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685307, 0.4688454
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2702550, 0.2705197
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692321, 0.2694268
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7039742, 0.7015901
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2827426, 0.2836068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3470609, upper bound: 0.2991537
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3639704, upper bound: 0.2983784
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4687667, 0.4689822
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706050, 0.2707848
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2695620, 0.2696955
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7069407, 0.7053151
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2831552, 0.2837455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2788344, upper bound: 0.2905904
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2746813, upper bound: 0.3028267
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688709, 0.4688668
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2706918, 0.2706877
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696260, 0.2696241
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7060642, 0.7061000
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2834409, 0.2834271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3004891, upper bound: 0.3443533
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2879838, upper bound: 0.3513121
time: 0.88 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3567588, upper bound: 0.2832159
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3399441, upper bound: 0.2838770
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3385485, upper bound: 0.2768956
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3458161, upper bound: 0.2769105
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2736200, upper bound: 0.3567433
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2747641, upper bound: 0.3574542
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2735499, upper bound: 0.3395827
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2742378, upper bound: 0.3508419
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3575048, upper bound: 0.2818219
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3568409, upper bound: 0.2810132
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3637466, upper bound: 0.2874453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3496709, upper bound: 0.2872339
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3543320, upper bound: 0.2771570
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3468271, upper bound: 0.2770626
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3430530, upper bound: 0.2838342
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3623051, upper bound: 0.2838145
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3557572, upper bound: 0.3129398
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3027823, upper bound: 0.3381835
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3534008, upper bound: 0.2894772
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3355848, upper bound: 0.2872739
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3531364, upper bound: 0.3089796
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3560222, upper bound: 0.3088822
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3674607, upper bound: 0.2940390
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3674003, upper bound: 0.2945333
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3151512, upper bound: 0.3555057
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3158696, upper bound: 0.3387697
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2922330, upper bound: 0.3477708
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2853519, upper bound: 0.3536985
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2868483, upper bound: 0.3419224
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2856043, upper bound: 0.3565154
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3072620, upper bound: 0.2770984
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2957362, upper bound: 0.2770984
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3400644, upper bound: 0.2907136
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3493125, upper bound: 0.2884243
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2857420, upper bound: 0.3387089
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2852545, upper bound: 0.3571104
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2722499, upper bound: 0.2892733
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2722499, upper bound: 0.3024423
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3496527, upper bound: 0.3289376
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3196271, upper bound: 0.3374940
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3270979, upper bound: 0.3452662
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3598361, upper bound: 0.3307681
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3132045, upper bound: 0.3441110
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3133884, upper bound: 0.3429460
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3452414, upper bound: 0.3214390
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3233361, upper bound: 0.3272876
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3643565, upper bound: 0.2899726
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3567080, upper bound: 0.2900102
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2853496, upper bound: 0.2916595
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2818134, upper bound: 0.3031622
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3166681, upper bound: 0.3477306
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3157185, upper bound: 0.3657510
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3074021, upper bound: 0.2942200
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2911563, upper bound: 0.3671405
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2921368, upper bound: 0.3557593
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2991503, upper bound: 0.3550005
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3433116, upper bound: 0.3230461
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3443787, upper bound: 0.3060311
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2961168, upper bound: 0.2683930
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2862082, upper bound: 0.2789445
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3091794, upper bound: 0.3488263
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3093525, upper bound: 0.3385325
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3160349, upper bound: 0.3545673
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3059343, upper bound: 0.3623549
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3602301, upper bound: 0.3106432
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3444597, upper bound: 0.3088049
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3661268, upper bound: 0.2875134
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2882173, upper bound: 0.2954512
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3572397, upper bound: 0.3065165
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3516968, upper bound: 0.3075627
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3470031, upper bound: 0.2820907
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3414256, upper bound: 0.2892342
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3556539, upper bound: 0.2812126
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2810871, upper bound: 0.3134068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3470609, upper bound: 0.2991537
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3639704, upper bound: 0.2983784
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2788344, upper bound: 0.2905904
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2746813, upper bound: 0.3028267
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.3004891, upper bound: 0.3443533
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.2879838, upper bound: 0.3513121

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4683861, 0.4684343
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700985, 0.2701391
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2690968, 0.2691264
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7002907, 0.6999257
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2824580, 0.2825902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3454411, upper bound: 0.2745729
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3358768, upper bound: 0.2721176
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690890, 0.4689493
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707777, 0.2706599
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696434, 0.2695574
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056360, 0.7067008
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836474, 0.2832627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2715993, upper bound: 0.3553484
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2715996, upper bound: 0.3355301
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692218, 0.4688659
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708893, 0.2705898
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697254, 0.2695057
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7050037, 0.7077074
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2840109, 0.2830346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2737371, upper bound: 0.3479888
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2730016, upper bound: 0.3567269
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690824, 0.4689498
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2707720, 0.2706605
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696392, 0.2695577
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056408, 0.7066493
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836291, 0.2832643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3524694, upper bound: 0.2708698
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3462015, upper bound: 0.2755605
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692206, 0.4688709
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708882, 0.2705941
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697247, 0.2695088
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7050428, 0.7076983
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2840077, 0.2830486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 142

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3431808, upper bound: 0.2712177
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3269675, upper bound: 0.2727553
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4691945, 0.4690082
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708178, 0.2706611
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2696257, 0.2695112
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7052221, 0.7066379
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843290, 0.2838182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3586629, upper bound: 0.2767218
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3523670, upper bound: 0.2812248
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4682618, 0.4685874
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2700902, 0.2703637
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2691096, 0.2693100
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7025318, 0.7000599
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2821313, 0.2830241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3615876, upper bound: 0.2826889
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3553344, upper bound: 0.2828389
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688113, 0.4689622
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705151, 0.2706420
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694294, 0.2695222
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7052507, 0.7041044
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2831013, 0.2835156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3537552, upper bound: 0.2814181
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2873989, upper bound: 0.3106211
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4690607, 0.4686854
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2708514, 0.2705349
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697431, 0.2695122
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7046843, 0.7075429
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2839619, 0.2829287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 177

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3022302, upper bound: 0.2756415
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2917636, upper bound: 0.2795640
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692570, 0.4690039
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709046, 0.2706919
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697124, 0.2695572
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7056465, 0.7075782
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843310, 0.2836310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3623574, upper bound: 0.2844547
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556419, upper bound: 0.2877242
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4692796, 0.4689884
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2709238, 0.2706789
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2697263, 0.2695475
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7055292, 0.7077494
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2843930, 0.2835884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3319090, upper bound: 0.2926443
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3658997, upper bound: 0.2925858
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688073, 0.4682617
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2705482, 0.2700902
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2694448, 0.2691095
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7000599, 0.7042131
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836264, 0.2821313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 116

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3006338, upper bound: 0.3431597
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3019326, upper bound: 0.3356989
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688102, 0.4681964
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2704559, 0.2699388
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693580, 0.2689798
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6984825, 0.7031596
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836258, 0.2819369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 248
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2826360, upper bound: 0.3541116
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2825114, upper bound: 0.3441285
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4688102, 0.4681964
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2704559, 0.2699388
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2693580, 0.2689798
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.6984825, 0.7031596
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2836258, 0.2819369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 142
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 140
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 121
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 177
type: RSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2669539, upper bound: 0.2852036
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2669539, upper bound: 0.2995892
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2387780, 0.2354041, -0.2387780, 0.2354041, -0.4685296, 0.4685426
1: -0.0994219, 0.1012199, -0.0994219, 0.1012199, -0.2006418, 0.2006418
2: -0.1509733, 0.1439491, -0.1509733, 0.1439491, -0.2949225, 0.2949225
3: -0.1204610, 0.1680746, -0.1204610, 0.1680746, -0.2885356, 0.2885356
4: -0.1072310, 0.0874664, -0.1072310, 0.0874664, -0.1946974, 0.1946974
5: -0.1294090, 0.1440715, -0.1294090, 0.1440715, -0.2703149, 0.2703261
6: -0.1511217, 0.1229078, -0.1511217, 0.1229078, -0.2692738, 0.2692823
7: -0.1029300, 0.1302250, -0.1029300, 0.1302250, -0.2331551, 0.2331551
8: 0.4503762, 1.1877248, 0.4503762, 1.1877248, -0.7021914, 0.7021022
9: -0.1230348, 0.1788786, -0.1230348, 0.1788786, -0.2828643, 0.2829012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.29 + 596.14 = 600.43 seconds
