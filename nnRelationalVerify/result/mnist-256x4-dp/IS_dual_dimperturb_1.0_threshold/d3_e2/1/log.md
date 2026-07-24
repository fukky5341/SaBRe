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
Threshold: 3.518859945


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=155, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398)
1: (-1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865)
2: (-1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576)
3: (-1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979)
4: (-1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516)
5: (-1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349)
6: (-1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487)
7: (-1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090)
8: (-2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340)
9: (-1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 3.72 = 5.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -3.7040631, upper bound: 3.7040631

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6751904, upper bound: 3.4297256
time: 1.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4279516, upper bound: 3.4279516
time: 0.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.94
Output dim: 8, lower bound: -3.6751904, upper bound: 3.4297256
IS_A2, status: Status.VERIFIED, split count: 1, time: 2.94
Output dim: 8, lower bound: -3.4279516, upper bound: 3.4279516

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -1.8331313, 1.3748745, -1.8503866, 1.3866532, -3.2197845, 3.2252612
1: -1.4452492, 1.3570296, -1.4591178, 1.3697687, -2.8150179, 2.8161473
2: -1.7716246, 1.9572396, -1.7938340, 1.9670236, -3.7386482, 3.7510736
3: -1.8107150, 1.2074822, -1.8323436, 1.2170542, -3.0277691, 3.0398259
4: -1.9427700, 1.4541137, -1.9638085, 1.4680431, -3.4108131, 3.4179223
5: -1.5599419, 1.4449782, -1.5762671, 1.4573678, -3.0173097, 3.0212455
6: -1.4364133, 1.6433718, -1.4516947, 1.6586541, -3.0950675, 3.0950665
7: -1.7823257, 1.6983439, -1.8006461, 1.7152629, -3.4975886, 3.4989901
8: -2.4092338, 1.6109619, -2.4382658, 1.6120684, -4.0213022, 4.0492277
9: -1.4043994, 1.6713778, -1.4212788, 1.6854837, -3.0898831, 3.0926566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=154, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.3490799, upper bound: 3.3501526
time: 1.44 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6506905, upper bound: 3.3945759
time: 1.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.45 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 4.45
Output dim: 8, lower bound: -3.3490799, upper bound: 3.3501526
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.45
Output dim: 8, lower bound: -3.6506905, upper bound: 3.3945759

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -1.5782301, 1.2034922, -1.8503866, 1.3866532, -2.9648833, 3.0538788
1: -1.2398256, 1.1726516, -1.4591178, 1.3697687, -2.6095943, 2.6317694
2: -1.4440213, 1.8074298, -1.7938340, 1.9670236, -3.4110451, 3.6012638
3: -1.4938318, 1.0716017, -1.8323436, 1.2170542, -2.7108860, 2.9039454
4: -1.6316648, 1.2559583, -1.9638085, 1.4680431, -3.0997078, 3.2197669
5: -1.3193698, 1.2698944, -1.5762671, 1.4573678, -2.7767377, 2.8461614
6: -1.2132719, 1.4216893, -1.4516947, 1.6586541, -2.8719258, 2.8733840
7: -1.5175573, 1.4598839, -1.8006461, 1.7152629, -3.2328200, 3.2605300
8: -2.0397341, 1.5992708, -2.4382658, 1.6120684, -3.6518025, 4.0375366
9: -1.1978384, 1.4625313, -1.4212788, 1.6854837, -2.8833222, 2.8838100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2475694, upper bound: 3.2329782
time: 1.22 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5077175, upper bound: 3.3298418
time: 1.48 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6504112, upper bound: 3.3940350
time: 2.41 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6502624, upper bound: 3.3940544
time: 1.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.02 seconds
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 8, lower bound: -3.6504112, upper bound: 3.3940350
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 8, lower bound: -3.6502624, upper bound: 3.3940544

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -1.4883584, 1.1435575, -1.8479636, 1.3849458, -2.8733041, 2.9915211
1: -1.1685778, 1.1059674, -1.4571939, 1.3679423, -2.5365200, 2.5631614
2: -1.3276401, 1.7556982, -1.7906122, 1.9655647, -3.2932048, 3.5463104
3: -1.3839824, 1.0233257, -1.8293526, 1.2156873, -2.5996697, 2.8526783
4: -1.5227377, 1.1862963, -1.9608924, 1.4660790, -2.9888167, 3.1471887
5: -1.2356305, 1.2063630, -1.5739666, 1.4555796, -2.6912103, 2.7803297
6: -1.1303564, 1.3453220, -1.4493721, 1.6565256, -2.7868819, 2.7946939
7: -1.4267834, 1.3769603, -1.7981067, 1.7128935, -3.1396770, 3.1750669
8: -1.9059148, 1.5681987, -2.4339812, 1.6112878, -3.5172014, 4.0021801
9: -1.1263269, 1.3869265, -1.4189128, 1.6834327, -2.8097596, 2.8058393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2408659, upper bound: 3.2305483
time: 1.46 seconds

## Relational analysis of IS_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
time: 1.63 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870056, upper bound: 3.3234557
time: 1.65 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -1.6837021, 1.2714183, -1.8411100, 1.3800902, -3.0637922, 3.1125283
1: -1.3256965, 1.2467302, -1.4517621, 1.3627757, -2.6884723, 2.6984923
2: -1.5766954, 1.8682749, -1.7814727, 1.9613113, -3.5380068, 3.6497476
3: -1.6255429, 1.1267391, -1.8209605, 1.2118137, -2.8373566, 2.9476995
4: -1.7613895, 1.3372796, -1.9526932, 1.4605322, -3.2219217, 3.2899728
5: -1.4184468, 1.3398876, -1.5674875, 1.4505038, -2.8689506, 2.9073751
6: -1.2988575, 1.5127316, -1.4427658, 1.6505036, -2.9493611, 2.9554973
7: -1.6265918, 1.5580024, -1.7909603, 1.7062262, -3.3328180, 3.3489628
8: -2.1867654, 1.5781059, -2.4216583, 1.6084380, -3.7952034, 3.9997642
9: -1.2800142, 1.5462757, -1.4122462, 1.6775661, -2.9575801, 2.9585218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=155, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2384783, upper bound: 3.2273243
time: 1.52 seconds

## Relational analysis of IS_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5075001, upper bound: 3.3293112
time: 1.26 seconds

## Relational analysis of IS_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6502624, upper bound: 3.3940544
time: 1.93 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.6502624, upper bound: 3.3940544
time: 1.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.72 seconds
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 8, lower bound: -3.5870056, upper bound: 3.3234557
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 8, lower bound: -3.6502624, upper bound: 3.3940544
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 8, lower bound: -3.6502624, upper bound: 3.3940544

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -1.4883584, 1.1435575, -1.5843379, 1.2070769, -2.6954353, 2.7278955
1: -1.1685778, 1.1059674, -1.2452757, 1.1756492, -2.3442268, 2.3512430
2: -1.3276401, 1.7556982, -1.4520785, 1.8156734, -3.1433134, 3.2077765
3: -1.3839824, 1.0233257, -1.5007362, 1.0743220, -2.4583044, 2.5240619
4: -1.5227377, 1.1862963, -1.6384919, 1.2603561, -2.7830939, 2.8247881
5: -1.2356305, 1.2063630, -1.3246323, 1.2730618, -2.5086923, 2.5309954
6: -1.1303564, 1.3453220, -1.2152846, 1.4273556, -2.5577121, 2.5606065
7: -1.4267834, 1.3769603, -1.5241052, 1.4653116, -2.8920951, 2.9010653
8: -1.9059148, 1.5681987, -2.0502853, 1.5912217, -3.4971359, 3.6184840
9: -1.1263269, 1.3869265, -1.2020931, 1.4669232, -2.5932503, 2.5890198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1508699, upper bound: 3.1509537
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870056, upper bound: 3.3234557
time: 1.24 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870056, upper bound: 3.3234557
time: 1.53 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -1.4299333, 1.1057230, -1.6519336, 1.2512523, -2.6811857, 2.7576566
1: -1.1216663, 1.0637623, -1.2999034, 1.2239957, -2.3456621, 2.3636656
2: -1.2536711, 1.7231276, -1.5366684, 1.8529768, -3.1066480, 3.2597961
3: -1.3115485, 0.9926741, -1.5846972, 1.1100508, -2.4215994, 2.5773714
4: -1.4512088, 1.1413844, -1.7214625, 1.3125437, -2.7637525, 2.8628469
5: -1.1809851, 1.1663655, -1.3879732, 1.3187289, -2.4997139, 2.5543387
6: -1.0795949, 1.2953618, -1.2720981, 1.4856632, -2.5652580, 2.5674598
7: -1.3671556, 1.3227608, -1.5934100, 1.5277901, -2.8949456, 2.9161708
8: -1.8228061, 1.5646999, -2.1489334, 1.5948093, -3.4176154, 3.7136333
9: -1.0803995, 1.3393017, -1.2549882, 1.5220201, -2.6024196, 2.5942898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=145, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1276959, upper bound: 3.1192048
time: 1.13 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
time: 1.64 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
time: 1.47 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6837021, 1.2714183, -1.7598157, 1.3221916, -3.0058937, 3.0312340
1: -1.3256965, 1.2467302, -1.3870904, 1.3014767, -2.6271732, 2.6338205
2: -1.5766954, 1.8682749, -1.6734719, 1.9129508, -3.4896462, 3.5417469
3: -1.6255429, 1.1267391, -1.7201092, 1.1671890, -2.7927318, 2.8468485
4: -1.7613895, 1.3372796, -1.8544021, 1.3962469, -3.1576364, 3.1916816
5: -1.4184468, 1.3398876, -1.4896853, 1.3918449, -2.8102918, 2.8295729
6: -1.2988575, 1.5127316, -1.3637000, 1.5789745, -2.8778319, 2.8764315
7: -1.6265918, 1.5580024, -1.7055047, 1.6288787, -3.2554705, 3.2635069
8: -2.1867654, 1.5781059, -2.2965367, 1.5798583, -3.7666237, 3.8746426
9: -1.2800142, 1.5462757, -1.3411598, 1.6081976, -2.8882117, 2.8874354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=152, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2384783, upper bound: 3.2273243
time: 1.18 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5075001, upper bound: 3.3293112
time: 1.95 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5926721, upper bound: 3.3658636
time: 1.53 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.57 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6837021, 1.2714183, -1.9679916, 1.4647675, -3.1484694, 3.2394099
1: -1.3256965, 1.2467302, -1.5545183, 1.4544364, -2.7801328, 2.8012486
2: -1.5766954, 1.8682749, -1.9433022, 2.0365756, -3.6132710, 3.8115771
3: -1.6255429, 1.1267391, -1.9801180, 1.2814881, -2.9070311, 3.1068573
4: -1.7613895, 1.3372796, -2.1073596, 1.5625219, -3.3239114, 3.4446392
5: -1.4184468, 1.3398876, -1.6864974, 1.5400167, -2.9584634, 3.0263851
6: -1.2988575, 1.5127316, -1.5484076, 1.7632661, -3.0621235, 3.0611391
7: -1.6265918, 1.5580024, -1.9265044, 1.8304660, -3.4570580, 3.4845066
8: -2.1867654, 1.5781059, -2.6304753, 1.5911393, -3.7779047, 4.2085810
9: -1.2800142, 1.5462757, -1.5361035, 1.7786212, -3.0586352, 3.0823793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=161, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2384783, upper bound: 3.2273243
time: 1.50 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.5075001, upper bound: 3.3293112
time: 1.47 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5926721, upper bound: 3.3658636
time: 1.33 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.45 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.20 seconds
IS_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5870056, upper bound: 3.3234557
IS_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5870056, upper bound: 3.3234557
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5926721, upper bound: 3.3658636
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5926721, upper bound: 3.3658636
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643

## BFS IS instance: IS_A1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.2318439, 0.9792398, -1.5843379, 1.2070769, -2.4389210, 2.5635777
1: -0.9627665, 0.9252325, -1.2452757, 1.1756492, -2.1384158, 2.1705084
2: -1.0127676, 1.6125829, -1.4520785, 1.8156734, -2.8284409, 3.0646615
3: -1.0669849, 0.8928139, -1.5007362, 1.0743220, -2.1413069, 2.3935502
4: -1.2079327, 0.9914424, -1.6384919, 1.2603561, -2.4682889, 2.6299343
5: -1.0007118, 1.0308343, -1.3246323, 1.2730618, -2.2737737, 2.3554666
6: -0.9122838, 1.1279083, -1.2152846, 1.4273556, -2.3396394, 2.3431931
7: -1.1678990, 1.1410156, -1.5241052, 1.4653116, -2.6332107, 2.6651208
8: -1.5463579, 1.5528561, -2.0502853, 1.5912217, -3.1375794, 3.6031413
9: -0.9279491, 1.1817434, -1.2020931, 1.4669232, -2.3948722, 2.3838365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1508699, upper bound: 3.1509537
time: 1.28 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
time: 1.56 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
time: 1.56 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.2950011, 1.0191262, -1.5843379, 1.2070769, -2.5020781, 2.6034641
1: -1.0136341, 0.9682163, -1.2452757, 1.1756492, -2.1892834, 2.2134919
2: -1.0876561, 1.6482435, -1.4520785, 1.8156734, -2.9033294, 3.1003220
3: -1.1446222, 0.9247266, -1.5007362, 1.0743220, -2.2189441, 2.4254627
4: -1.2856369, 1.0392070, -1.6384919, 1.2603561, -2.5459929, 2.6776989
5: -1.0579793, 1.0735079, -1.3246323, 1.2730618, -2.3310411, 2.3981402
6: -0.9645361, 1.1815189, -1.2152846, 1.4273556, -2.3918917, 2.3968034
7: -1.2315900, 1.1980875, -1.5241052, 1.4653116, -2.6969018, 2.7221928
8: -1.6361153, 1.5560350, -2.0502853, 1.5912217, -3.2273369, 3.6063204
9: -0.9756680, 1.2316275, -1.2020931, 1.4669232, -2.4425912, 2.4337206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1508699, upper bound: 3.1509537
time: 1.17 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
time: 1.55 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
time: 1.52 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.4299333, 1.1057230, -1.5636833, 1.1918125, -2.6217458, 2.6694064
1: -1.1216663, 1.0637623, -1.2299134, 1.1585059, -2.2801723, 2.2936757
2: -1.2536711, 1.7231276, -1.4223560, 1.8026776, -3.0563488, 3.1454835
3: -1.3115485, 0.9926741, -1.4766717, 1.0621344, -2.3736830, 2.4693458
4: -1.4512088, 1.1413844, -1.6143688, 1.2437297, -2.6949387, 2.7557530
5: -1.1809851, 1.1663655, -1.3055284, 1.2566075, -2.4375925, 2.4718938
6: -1.0795949, 1.2953618, -1.1911466, 1.4105426, -2.4901376, 2.4865084
7: -1.3671556, 1.3227608, -1.5040827, 1.4463542, -2.8135097, 2.8268435
8: -1.8228061, 1.5646999, -2.0174265, 1.5664601, -3.3892663, 3.5821264
9: -1.0803995, 1.3393017, -1.1847454, 1.4479086, -2.5283082, 2.5240471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=145, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1276959, upper bound: 3.1192048
time: 1.16 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
time: 1.74 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
time: 1.32 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.4299333, 1.1057230, -1.7665946, 1.3251516, -2.7550850, 2.8723178
1: -1.1216663, 1.0637623, -1.3931432, 1.3046700, -2.4263363, 2.4569054
2: -1.2536711, 1.7231276, -1.6809403, 1.9194891, -3.1731601, 3.4040680
3: -1.3115485, 0.9926741, -1.7276958, 1.1700671, -2.4816155, 2.7203698
4: -1.4512088, 1.1413844, -1.8623216, 1.4009686, -2.8521774, 3.0037060
5: -1.1809851, 1.1663655, -1.4956250, 1.3949476, -2.5759327, 2.6619906
6: -1.0795949, 1.2953618, -1.3656166, 1.5845485, -2.6641433, 2.6609783
7: -1.3671556, 1.3227608, -1.7117693, 1.6343319, -3.0014875, 3.0345302
8: -1.8228061, 1.5646999, -2.3090327, 1.5742934, -3.3970995, 3.8737326
9: -1.0803995, 1.3393017, -1.3443698, 1.6131978, -2.6935973, 2.6836715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=145, inp2_unstable=152, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1276959, upper bound: 3.1192048
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4281984, upper bound: 3.2563000
time: 1.55 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
time: 1.45 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
time: 1.42 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -1.6837021, 1.2714183, -1.4999933, 1.1508595, -2.8345616, 2.7714117
1: -1.3256965, 1.2467302, -1.1784400, 1.1130669, -2.4387634, 2.4251702
2: -1.5766954, 1.8682749, -1.3432245, 1.7677634, -3.3444588, 3.2114995
3: -1.6255429, 1.1267391, -1.3975259, 1.0290018, -2.6545448, 2.5242651
4: -1.7613895, 1.3372796, -1.5361704, 1.1950091, -2.9563985, 2.8734498
5: -1.4184468, 1.3398876, -1.2459900, 1.2135566, -2.6320033, 2.5858777
6: -1.2988575, 1.5127316, -1.1376295, 1.3556755, -2.6545329, 2.6503611
7: -1.6265918, 1.5580024, -1.4389282, 1.3875118, -3.0141037, 2.9969306
8: -2.1867654, 1.5781059, -1.9252994, 1.5629038, -3.7496691, 3.5034051
9: -1.2800142, 1.5462757, -1.1350180, 1.3960116, -2.6760259, 2.6812937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=147, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1515745, upper bound: 3.1452932
time: 1.36 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.27 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.53 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -1.6248791, 1.2326597, -1.5636833, 1.1918125, -2.8166916, 2.7963428
1: -1.2784715, 1.2041166, -1.2299134, 1.1585059, -2.4369774, 2.4340301
2: -1.5017555, 1.8352771, -1.4223560, 1.8026776, -3.3044331, 3.2576332
3: -1.5526688, 1.0953553, -1.4766717, 1.0621344, -2.6148033, 2.5720270
4: -1.6894015, 1.2916363, -1.6143688, 1.2437297, -2.9331312, 2.9060051
5: -1.3632683, 1.2995884, -1.3055284, 1.2566075, -2.6198759, 2.6051168
6: -1.2477161, 1.4623556, -1.1911466, 1.4105426, -2.6582587, 2.6535022
7: -1.5664328, 1.5034585, -1.5040827, 1.4463542, -3.0127869, 3.0075412
8: -2.1024432, 1.5740025, -2.0174265, 1.5664601, -3.6689034, 3.5914290
9: -1.2336334, 1.4982324, -1.1847454, 1.4479086, -2.6815419, 2.6829777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1276442, upper bound: 3.1152574
time: 1.26 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.34 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.31 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1.6837021, 1.2714183, -1.7146471, 1.2914681, -2.9751701, 2.9860654
1: -1.3256965, 1.2467302, -1.3511385, 1.2676022, -2.5932987, 2.5978687
2: -1.5766954, 1.8682749, -1.6166415, 1.8913424, -3.4680378, 3.4849164
3: -1.6255429, 1.1267391, -1.6631278, 1.1427015, -2.7682443, 2.7898669
4: -1.7613895, 1.3372796, -1.7984920, 1.3609711, -3.1223607, 3.1357715
5: -1.4184468, 1.3398876, -1.4469213, 1.3600681, -2.7785149, 2.7868090
6: -1.2988575, 1.5127316, -1.3223453, 1.5397336, -2.8385911, 2.8350768
7: -1.6265918, 1.5580024, -1.6585599, 1.5864944, -3.2130861, 3.2165623
8: -2.1867654, 1.5781059, -2.2329381, 1.5720397, -3.7588053, 3.8110440
9: -1.2800142, 1.5462757, -1.3038367, 1.5709574, -2.8509717, 2.8501124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=150, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1478066, upper bound: 3.1437612
time: 1.27 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.44 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.33 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1.6248791, 1.2326597, -1.7667954, 1.3263429, -2.9512219, 2.9994550
1: -1.2784715, 1.2041166, -1.3933798, 1.3051699, -2.5836415, 2.5974965
2: -1.5017555, 1.8352771, -1.6811196, 1.9202919, -3.4220474, 3.5163965
3: -1.5526688, 1.0953553, -1.7284360, 1.1702833, -2.7229521, 2.8237913
4: -1.6894015, 1.2916363, -1.8625708, 1.4012654, -3.0906668, 3.1542072
5: -1.3632683, 1.2995884, -1.4957174, 1.3955950, -2.7588632, 2.7953057
6: -1.2477161, 1.4623556, -1.3657775, 1.5858554, -2.8335714, 2.8281331
7: -1.5664328, 1.5034585, -1.7132732, 1.6348935, -3.2013264, 3.2167315
8: -2.1024432, 1.5740025, -2.3096759, 1.5748332, -3.6772764, 3.8836784
9: -1.2336334, 1.4982324, -1.3465314, 1.6133627, -2.8469961, 2.8447638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=148, inp2_unstable=152, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1249276, upper bound: 3.1140373
time: 1.58 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4271918, upper bound: 3.2556783
time: 1.49 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.80 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
time: 1.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 20.30 seconds
IS_A1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
IS_A1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
IS_A1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
IS_A1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5927040, upper bound: 3.3658624
IS_A1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
IS_A1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
IS_A1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
IS_A1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234557
IS_A1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643
IS_A1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 20.30
Output dim: 8, lower bound: -3.5870018, upper bound: 3.3234643

## BFS IS instance: IS_A1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.2318439, 0.9792398, -1.4999933, 1.1508595, -2.3827033, 2.4792333
1: -0.9627665, 0.9252325, -1.1784400, 1.1130669, -2.0758333, 2.1036725
2: -1.0127676, 1.6125829, -1.3432245, 1.7677634, -2.7805309, 2.9558074
3: -1.0669849, 0.8928139, -1.3975259, 1.0290018, -2.0959868, 2.2903399
4: -1.2079327, 0.9914424, -1.5361704, 1.1950091, -2.4029417, 2.5276127
5: -1.0007118, 1.0308343, -1.2459900, 1.2135566, -2.2142684, 2.2768245
6: -0.9122838, 1.1279083, -1.1376295, 1.3556755, -2.2679591, 2.2655377
7: -1.1678990, 1.1410156, -1.4389282, 1.3875118, -2.5554109, 2.5799439
8: -1.5463579, 1.5528561, -1.9252994, 1.5629038, -3.1092615, 3.4781556
9: -0.9279491, 1.1817434, -1.1350180, 1.3960116, -2.3239608, 2.3167615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=147, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2107418, upper bound: 3.2004746
time: 1.32 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4789169, upper bound: 3.3009227
time: 1.29 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9060794, upper bound: 2.9010071
time: 1.19 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4921554, upper bound: 3.3053807
time: 1.24 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4893407, upper bound: 3.2503603
time: 1.39 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.2318439, 0.9792398, -1.7145182, 1.2912898, -2.5231338, 2.6937580
1: -0.9627665, 0.9252325, -1.3510562, 1.2674766, -2.2302432, 2.2762887
2: -1.0127676, 1.6125829, -1.6164609, 1.8912416, -2.9040091, 3.2290440
3: -1.0669849, 0.8928139, -1.6629491, 1.1426190, -2.2096038, 2.5557630
4: -1.2079327, 0.9914424, -1.7983577, 1.3608761, -2.5688088, 2.7898002
5: -1.0007118, 1.0308343, -1.4468265, 1.3599197, -2.3606315, 2.4776607
6: -0.9122838, 1.1279083, -1.3221347, 1.5395725, -2.4518561, 2.4500432
7: -1.1678990, 1.1410156, -1.6584723, 1.5863605, -2.7542596, 2.7994881
8: -1.5463579, 1.5528561, -2.2325447, 1.5713158, -3.1176732, 3.7854009
9: -0.9279491, 1.1817434, -1.3036833, 1.5707980, -2.4987471, 2.4854267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=150, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.2107418, upper bound: 3.2004746
time: 1.19 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4789169, upper bound: 3.3009227
time: 1.32 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.9060794, upper bound: 2.9010071
time: 1.12 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4921554, upper bound: 3.3053807
time: 2.04 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4893407, upper bound: 3.2503603
time: 1.41 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.2950011, 1.0191262, -1.4999933, 1.1508595, -2.4458606, 2.5191195
1: -1.0136341, 0.9682163, -1.1784400, 1.1130669, -2.1267009, 2.1466563
2: -1.0876561, 1.6482435, -1.3432245, 1.7677634, -2.8554196, 2.9914680
3: -1.1446222, 0.9247266, -1.3975259, 1.0290018, -2.1736240, 2.3222525
4: -1.2856369, 1.0392070, -1.5361704, 1.1950091, -2.4806461, 2.5753775
5: -1.0579793, 1.0735079, -1.2459900, 1.2135566, -2.2715359, 2.3194981
6: -0.9645361, 1.1815189, -1.1376295, 1.3556755, -2.3202114, 2.3191485
7: -1.2315900, 1.1980875, -1.4389282, 1.3875118, -2.6191020, 2.6370158
8: -1.6361153, 1.5560350, -1.9252994, 1.5629038, -3.1990187, 3.4813342
9: -0.9756680, 1.2316275, -1.1350180, 1.3960116, -2.3716795, 2.3666453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=147, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1508699, upper bound: 3.1509537
time: 1.32 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4413108, upper bound: 3.2970384
time: 1.19 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4702951, upper bound: 3.3043927
time: 1.31 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4671083, upper bound: 3.2496749
time: 1.42 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.2950011, 1.0191262, -1.7145182, 1.2912898, -2.5862908, 2.7336445
1: -1.0136341, 0.9682163, -1.3510562, 1.2674766, -2.2811108, 2.3192725
2: -1.0876561, 1.6482435, -1.6164609, 1.8912416, -2.9788976, 3.2647045
3: -1.1446222, 0.9247266, -1.6629491, 1.1426190, -2.2872412, 2.5876756
4: -1.2856369, 1.0392070, -1.7983577, 1.3608761, -2.6465130, 2.8375647
5: -1.0579793, 1.0735079, -1.4468265, 1.3599197, -2.4178991, 2.5203342
6: -0.9645361, 1.1815189, -1.3221347, 1.5395725, -2.5041084, 2.5036535
7: -1.2315900, 1.1980875, -1.6584723, 1.5863605, -2.8179505, 2.8565598
8: -1.6361153, 1.5560350, -2.2325447, 1.5713158, -3.2074311, 3.7885797
9: -0.9756680, 1.2316275, -1.3036833, 1.5707980, -2.5464659, 2.5353107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=150, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1508699, upper bound: 3.1509537
time: 1.27 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4413108, upper bound: 3.2970384
time: 1.75 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4702951, upper bound: 3.3043927
time: 1.71 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4671083, upper bound: 3.2496749
time: 1.48 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.2318439, 0.9792398, -1.5636833, 1.1918125, -2.4236565, 2.5429230
1: -0.9627665, 0.9252325, -1.2299134, 1.1585059, -2.1212726, 2.1551459
2: -1.0127676, 1.6125829, -1.4223560, 1.8026776, -2.8154452, 3.0349388
3: -1.0669849, 0.8928139, -1.4766717, 1.0621344, -2.1291194, 2.3694856
4: -1.2079327, 0.9914424, -1.6143688, 1.2437297, -2.4516625, 2.6058111
5: -1.0007118, 1.0308343, -1.3055284, 1.2566075, -2.2573195, 2.3363628
6: -0.9122838, 1.1279083, -1.1911466, 1.4105426, -2.3228264, 2.3190551
7: -1.1678990, 1.1410156, -1.5040827, 1.4463542, -2.6142530, 2.6450982
8: -1.5463579, 1.5528561, -2.0174265, 1.5664601, -3.1128173, 3.5702825
9: -0.9279491, 1.1817434, -1.1847454, 1.4479086, -2.3758578, 2.3664889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1354537, upper bound: 3.1234235
time: 1.29 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4289391, upper bound: 3.2563000
time: 1.58 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4634459, upper bound: 3.2598900
time: 1.27 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4613911, upper bound: 3.2218886
time: 1.34 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.2950011, 1.0191262, -1.5636833, 1.1918125, -2.4868135, 2.5828094
1: -1.0136341, 0.9682163, -1.2299134, 1.1585059, -2.1721401, 2.1981297
2: -1.0876561, 1.6482435, -1.4223560, 1.8026776, -2.8903337, 3.0705996
3: -1.1446222, 0.9247266, -1.4766717, 1.0621344, -2.2067566, 2.4013982
4: -1.2856369, 1.0392070, -1.6143688, 1.2437297, -2.5293665, 2.6535759
5: -1.0579793, 1.0735079, -1.3055284, 1.2566075, -2.3145869, 2.3790364
6: -0.9645361, 1.1815189, -1.1911466, 1.4105426, -2.3750787, 2.3726654
7: -1.2315900, 1.1980875, -1.5040827, 1.4463542, -2.6779442, 2.7021701
8: -1.6361153, 1.5560350, -2.0174265, 1.5664601, -3.2025752, 3.5734615
9: -0.9756680, 1.2316275, -1.1847454, 1.4479086, -2.4235766, 2.4163728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1354537, upper bound: 3.1234235
time: 1.29 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4289391, upper bound: 3.2563000
time: 1.25 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4634459, upper bound: 3.2598900
time: 1.53 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4613911, upper bound: 3.2218886
time: 1.96 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2318439, 0.9792398, -1.7665946, 1.3251516, -2.5569954, 2.7458344
1: -0.9627665, 0.9252325, -1.3931432, 1.3046700, -2.2674365, 2.3183756
2: -1.0127676, 1.6125829, -1.6809403, 1.9194891, -2.9322567, 3.2935233
3: -1.0669849, 0.8928139, -1.7276958, 1.1700671, -2.2370520, 2.6205096
4: -1.2079327, 0.9914424, -1.8623216, 1.4009686, -2.6089013, 2.8537641
5: -1.0007118, 1.0308343, -1.4956250, 1.3949476, -2.3956594, 2.5264592
6: -0.9122838, 1.1279083, -1.3656166, 1.5845485, -2.4968324, 2.4935250
7: -1.1678990, 1.1410156, -1.7117693, 1.6343319, -2.8022308, 2.8527851
8: -1.5463579, 1.5528561, -2.3090327, 1.5742934, -3.1206512, 3.8618889
9: -0.9279491, 1.1817434, -1.3443698, 1.6131978, -2.5411468, 2.5261130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=134, inp2_unstable=152, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1276959, upper bound: 3.1192048
time: 1.32 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4281984, upper bound: 3.2563000
time: 1.30 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4634315, upper bound: 3.2598900
time: 1.48 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4611852, upper bound: 3.2218886
time: 1.82 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.2950011, 1.0191262, -1.7665946, 1.3251516, -2.6201527, 2.7857208
1: -1.0136341, 0.9682163, -1.3931432, 1.3046700, -2.3183041, 2.3613596
2: -1.0876561, 1.6482435, -1.6809403, 1.9194891, -3.0071454, 3.3291838
3: -1.1446222, 0.9247266, -1.7276958, 1.1700671, -2.3146892, 2.6524224
4: -1.2856369, 1.0392070, -1.8623216, 1.4009686, -2.6866055, 2.9015286
5: -1.0579793, 1.0735079, -1.4956250, 1.3949476, -2.4529271, 2.5691328
6: -0.9645361, 1.1815189, -1.3656166, 1.5845485, -2.5490847, 2.5471354
7: -1.2315900, 1.1980875, -1.7117693, 1.6343319, -2.8659220, 2.9098568
8: -1.6361153, 1.5560350, -2.3090327, 1.5742934, -3.2104087, 3.8650677
9: -0.9756680, 1.2316275, -1.3443698, 1.6131978, -2.5888658, 2.5759974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=135, inp2_unstable=152, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1276959, upper bound: 3.1192048
time: 1.15 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4281984, upper bound: 3.2563000
time: 1.83 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4634315, upper bound: 3.2598900
time: 1.40 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4611852, upper bound: 3.2218886
time: 1.48 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -1.4333173, 1.1077571, -1.4999933, 1.1508595, -2.5841768, 2.6077504
1: -1.1245995, 1.0655322, -1.1784400, 1.1130669, -2.2376664, 2.2439723
2: -1.2586958, 1.7289393, -1.3432245, 1.7677634, -3.0264592, 3.0721638
3: -1.3150125, 0.9943793, -1.3975259, 1.0290018, -2.3440142, 2.3919053
4: -1.4546192, 1.1440135, -1.5361704, 1.1950091, -2.6496282, 2.6801839
5: -1.1839325, 1.1682850, -1.2459900, 1.2135566, -2.3974891, 2.4142752
6: -1.0814230, 1.2983909, -1.1376295, 1.3556755, -2.4370985, 2.4360204
7: -1.3706899, 1.3256383, -1.4389282, 1.3875118, -2.7582016, 2.7645664
8: -1.8299627, 1.5612292, -1.9252994, 1.5629038, -3.3928666, 3.4865284
9: -1.0828824, 1.3416806, -1.1350180, 1.3960116, -2.4788940, 2.4766986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=145, inp2_unstable=147, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1515745, upper bound: 3.1452932
time: 1.35 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4410357, upper bound: 3.2963485
time: 1.60 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4703772, upper bound: 3.3046189
time: 1.38 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4671794, upper bound: 3.2497687
time: 1.30 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1.4869409, 1.1421839, -1.4999933, 1.1508595, -2.6378002, 2.6421771
1: -1.1680306, 1.1036148, -1.1784400, 1.1130669, -2.2810974, 2.2820549
2: -1.3251060, 1.7585566, -1.3432245, 1.7677634, -3.0928693, 3.1017811
3: -1.3817090, 1.0219705, -1.3975259, 1.0290018, -2.4107108, 2.4194965
4: -1.5205135, 1.1848146, -1.5361704, 1.1950091, -2.7155228, 2.7209849
5: -1.2338409, 1.2044482, -1.2459900, 1.2135566, -2.4473977, 2.4504383
6: -1.1257464, 1.3446945, -1.1376295, 1.3556755, -2.4814219, 2.4823241
7: -1.4255490, 1.3751658, -1.4389282, 1.3875118, -2.8130608, 2.8140941
8: -1.9075460, 1.5638537, -1.9252994, 1.5629038, -3.4704499, 3.4891531
9: -1.1246711, 1.3853395, -1.1350180, 1.3960116, -2.5206828, 2.5203576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=147, inp2_unstable=147, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1515745, upper bound: 3.1452932
time: 1.36 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4410357, upper bound: 3.2963485
time: 1.36 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4703772, upper bound: 3.3046189
time: 1.62 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4671794, upper bound: 3.2497687
time: 1.36 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -1.4333173, 1.1077571, -1.5636833, 1.1918125, -2.6251297, 2.6714404
1: -1.1245995, 1.0655322, -1.2299134, 1.1585059, -2.2831054, 2.2954454
2: -1.2586958, 1.7289393, -1.4223560, 1.8026776, -3.0613735, 3.1512952
3: -1.3150125, 0.9943793, -1.4766717, 1.0621344, -2.3771467, 2.4710510
4: -1.4546192, 1.1440135, -1.6143688, 1.2437297, -2.6983490, 2.7583823
5: -1.1839325, 1.1682850, -1.3055284, 1.2566075, -2.4405401, 2.4738135
6: -1.0814230, 1.2983909, -1.1911466, 1.4105426, -2.4919658, 2.4895375
7: -1.3706899, 1.3256383, -1.5040827, 1.4463542, -2.8170440, 2.8297210
8: -1.8299627, 1.5612292, -2.0174265, 1.5664601, -3.3964229, 3.5786557
9: -1.0828824, 1.3416806, -1.1847454, 1.4479086, -2.5307910, 2.5264261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=145, inp2_unstable=148, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.1276442, upper bound: 3.1152574
time: 1.38 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.4275325, upper bound: 3.2556783
time: 1.52 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 5.19 + 595.12 = 600.31 seconds
