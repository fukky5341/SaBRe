## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00088668


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0006465, 0.0006465)
1: (0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000934, 0.0000934)
2: (0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003574, 0.0003574)
3: (-0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003697, 0.0003697)
4: (-0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0004002, 0.0004002)
5: (0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003787, 0.0003787)
6: (0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0015027, 0.0015027)
7: (-0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0020465, 0.0020465)
8: (0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0014416, 0.0014416)
9: (-0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0013086, 0.0013086)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 1.28 = 2.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0011530, upper bound: 0.0011530

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009740, upper bound: 0.0009740
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009740, upper bound: 0.0009740
time: 0.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 8, lower bound: -0.0009740, upper bound: 0.0009740
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 8, lower bound: -0.0009740, upper bound: 0.0009740

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0006458, 0.0006457
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000933, 0.0000933
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003570, 0.0003571
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003692, 0.0003693
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003998, 0.0003997
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003782, 0.0003783
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0015007, 0.0015011
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0020444, 0.0020439
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0014401, 0.0014397
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0013069, 0.0013072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008933, upper bound: 0.0008930
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008930, upper bound: 0.0008933
time: 0.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0006465, 0.0006458
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000934, 0.0000933
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003571, 0.0003574
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003693, 0.0003697
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0004002, 0.0003998
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003783, 0.0003787
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0015011, 0.0015027
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0020465, 0.0020444
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0014416, 0.0014401
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0013072, 0.0013086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008933, upper bound: 0.0008930
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008930, upper bound: 0.0008933
time: 0.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0008933, upper bound: 0.0008930
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0008930, upper bound: 0.0008933
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0008933, upper bound: 0.0008930
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0008930, upper bound: 0.0008933

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005565, 0.0005561
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000804, 0.0000803
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003075, 0.0003077
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003180, 0.0003182
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003445, 0.0003443
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003258, 0.0003260
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012926, 0.0012934
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017615, 0.0017604
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012408, 0.0012401
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011257, 0.0011263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008873, upper bound: 0.0008696
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008731, upper bound: 0.0008870
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005563, 0.0005566
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000804, 0.0000804
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003078, 0.0003076
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003183, 0.0003181
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003444, 0.0003446
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003261, 0.0003259
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012938, 0.0012930
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017610, 0.0017621
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012405, 0.0012412
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011267, 0.0011260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008870, upper bound: 0.0008731
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008696, upper bound: 0.0008873
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005569, 0.0005563
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000805, 0.0000804
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003076, 0.0003079
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003181, 0.0003185
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003448, 0.0003444
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003259, 0.0003263
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012930, 0.0012945
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017630, 0.0017610
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012419, 0.0012405
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011260, 0.0011273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008873, upper bound: 0.0008696
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008731, upper bound: 0.0008870
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005568, 0.0005565
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000804, 0.0000804
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003077, 0.0003078
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003182, 0.0003184
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003447, 0.0003445
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003260, 0.0003262
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012934, 0.0012941
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017625, 0.0017615
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012415, 0.0012408
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011263, 0.0011270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008870, upper bound: 0.0008731
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008696, upper bound: 0.0008873
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.37 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008873, upper bound: 0.0008696
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008731, upper bound: 0.0008870
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008870, upper bound: 0.0008731
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008696, upper bound: 0.0008873
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008873, upper bound: 0.0008696
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008731, upper bound: 0.0008870
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008870, upper bound: 0.0008731
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 8, lower bound: -0.0008696, upper bound: 0.0008873

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005549, 0.0005523
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000802, 0.0000798
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003053, 0.0003068
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003158, 0.0003173
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003435, 0.0003419
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003235, 0.0003250
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012836, 0.0012897
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017564, 0.0017482
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012373, 0.0012315
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011179, 0.0011231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005531, 0.0005546
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000799, 0.0000801
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003066, 0.0003058
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003171, 0.0003163
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003424, 0.0003433
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003249, 0.0003240
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012889, 0.0012856
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017509, 0.0017554
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012334, 0.0012366
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011225, 0.0011196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005547, 0.0005533
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000801, 0.0000799
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003059, 0.0003067
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003164, 0.0003172
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003434, 0.0003425
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003241, 0.0003250
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012860, 0.0012893
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017559, 0.0017515
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012369, 0.0012338
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011199, 0.0011228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005524, 0.0005551
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000798, 0.0000802
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003069, 0.0003054
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003174, 0.0003159
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003420, 0.0003436
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003252, 0.0003236
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012901, 0.0012840
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017487, 0.0017570
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012318, 0.0012377
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011235, 0.0011182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005554, 0.0005524
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000802, 0.0000798
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003054, 0.0003071
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003159, 0.0003176
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003438, 0.0003420
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003236, 0.0003254
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012840, 0.0012909
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017581, 0.0017487
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012384, 0.0012318
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011182, 0.0011242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005536, 0.0005547
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000800, 0.0000801
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003067, 0.0003061
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003172, 0.0003166
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003427, 0.0003434
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003250, 0.0003243
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012893, 0.0012868
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017525, 0.0017559
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012345, 0.0012369
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011228, 0.0011206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005552, 0.0005531
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000802, 0.0000799
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003058, 0.0003070
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003163, 0.0003175
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003437, 0.0003424
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003240, 0.0003253
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012856, 0.0012905
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017576, 0.0017509
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012381, 0.0012334
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011196, 0.0011238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0011797, 0.0021911, 0.0011797, 0.0021911, -0.0005530, 0.0005549
1: 0.0014927, 0.0016389, 0.0014927, 0.0016389, -0.0000799, 0.0000802
2: 0.0131484, 0.0137077, 0.0131484, 0.0137077, -0.0003068, 0.0003057
3: -0.0010817, -0.0005034, -0.0010817, -0.0005034, -0.0003173, 0.0003162
4: -0.0034920, -0.0028659, -0.0034920, -0.0028659, -0.0003423, 0.0003435
5: 0.0068255, 0.0074180, 0.0068255, 0.0074180, -0.0003250, 0.0003239
6: 0.0047812, 0.0071321, 0.0047812, 0.0071321, -0.0012897, 0.0012852
7: -0.0122701, -0.0090683, -0.0122701, -0.0090683, -0.0017504, 0.0017564
8: 0.9805705, 0.9828260, 0.9805705, 0.9828260, -0.0012330, 0.0012373
9: -0.0002978, 0.0017495, -0.0002978, 0.0017495, -0.0011231, 0.0011192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928
time: 0.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007928, upper bound: 0.0007684
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007747, upper bound: 0.0007915
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007915, upper bound: 0.0007747
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 8, lower bound: -0.0007684, upper bound: 0.0007928

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.88 + 96.23 = 99.11 seconds
