## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.002363328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232)
1: (0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496)
2: (-0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058373, 0.0058373)
3: (0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114)
4: (0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320)
5: (0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132)
6: (-0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944)
7: (-0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938)
8: (0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106174, 0.0106174)
9: (-0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 1.98 = 3.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0024712, upper bound: 0.0024711

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024395, upper bound: 0.0024398
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024398, upper bound: 0.0024392
time: 0.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.98 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.98
Output dim: 1, lower bound: -0.0024395, upper bound: 0.0024398
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.98
Output dim: 1, lower bound: -0.0024398, upper bound: 0.0024392

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058190, 0.0058261
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106146, 0.0106127
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024312, upper bound: 0.0024306
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024297, upper bound: 0.0024317
time: 1.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058373, 0.0058190
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106127, 0.0106174
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024327, upper bound: 0.0024318
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024327, upper bound: 0.0024319
time: 1.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 1, lower bound: -0.0024312, upper bound: 0.0024306
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 1, lower bound: -0.0024297, upper bound: 0.0024317
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 1, lower bound: -0.0024327, upper bound: 0.0024318
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 1, lower bound: -0.0024327, upper bound: 0.0024319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058028, 0.0058109
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106107, 0.0106085
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024241, upper bound: 0.0024268
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024273, upper bound: 0.0024242
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058038, 0.0058093
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106103, 0.0106088
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024222, upper bound: 0.0024246
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024223, upper bound: 0.0024246
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058277, 0.0058105
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106105, 0.0106149
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024149, upper bound: 0.0024146
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024149, upper bound: 0.0024144
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058288, 0.0058095
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106102, 0.0106152
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024313, upper bound: 0.0024250
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024262, upper bound: 0.0024304
time: 1.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024241, upper bound: 0.0024268
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024273, upper bound: 0.0024242
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024222, upper bound: 0.0024246
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024223, upper bound: 0.0024246
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024149, upper bound: 0.0024146
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024149, upper bound: 0.0024144
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024313, upper bound: 0.0024250
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 1, lower bound: -0.0024262, upper bound: 0.0024304

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057963, 0.0058103
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106118, 0.0106081
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024248, upper bound: 0.0024265
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024245, upper bound: 0.0024267
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058021, 0.0058035
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106100, 0.0106096
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024186, upper bound: 0.0024147
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024186, upper bound: 0.0024147
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057942, 0.0058001
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106079, 0.0106063
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024167, upper bound: 0.0024205
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024167, upper bound: 0.0024179
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057953, 0.0057997
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106078, 0.0106066
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024210, upper bound: 0.0024178
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024167, upper bound: 0.0024233
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058180, 0.0057993
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106074, 0.0106121
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024143
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024145
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058165, 0.0058105
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106105, 0.0106117
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024142
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024143
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058146, 0.0057896
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106044, 0.0106108
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023827, upper bound: 0.0023802
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023827, upper bound: 0.0023802
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058090, 0.0057952
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106059, 0.0106094
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024198, upper bound: 0.0024264
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024221, upper bound: 0.0024243
time: 1.25 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024248, upper bound: 0.0024265
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024245, upper bound: 0.0024267
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024186, upper bound: 0.0024147
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024186, upper bound: 0.0024147
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024167, upper bound: 0.0024205
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024167, upper bound: 0.0024179
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024210, upper bound: 0.0024178
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024167, upper bound: 0.0024233
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024143
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024145
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024142
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024147, upper bound: 0.0024143
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0023827, upper bound: 0.0023802
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0023827, upper bound: 0.0023802
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024198, upper bound: 0.0024264
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 1, lower bound: -0.0024221, upper bound: 0.0024243

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057962, 0.0058103
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106118, 0.0106081
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024016, upper bound: 0.0024043
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024018, upper bound: 0.0024043
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057962, 0.0058102
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106118, 0.0106081
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024105, upper bound: 0.0024174
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024152, upper bound: 0.0024120
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057854, 0.0057869
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106054, 0.0106050
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023607, upper bound: 0.0023607
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023607, upper bound: 0.0023607
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057867, 0.0057867
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106053, 0.0106053
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023971, upper bound: 0.0023935
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023971, upper bound: 0.0023937
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057877, 0.0057994
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106090, 0.0106059
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024100, upper bound: 0.0024136
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024100, upper bound: 0.0024193
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057936, 0.0057931
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106073, 0.0106074
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024103, upper bound: 0.0024102
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024104, upper bound: 0.0024091
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057815, 0.0057804
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106020, 0.0106023
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024075, upper bound: 0.0024091
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024031, upper bound: 0.0024033
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057760, 0.0057880
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106040, 0.0106008
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024166, upper bound: 0.0024231
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024166, upper bound: 0.0024232
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058177, 0.0057991
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106073, 0.0106121
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024135, upper bound: 0.0024094
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024101, upper bound: 0.0024129
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058176, 0.0057991
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106073, 0.0106120
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024071, upper bound: 0.0024106
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024071, upper bound: 0.0024072
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058163, 0.0058106
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106105, 0.0106117
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024135, upper bound: 0.0024094
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024101, upper bound: 0.0024129
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058162, 0.0058105
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106105, 0.0106117
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024071, upper bound: 0.0024106
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024108, upper bound: 0.0024072
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058053, 0.0057778
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106012, 0.0106082
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023489, upper bound: 0.0023462
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023489, upper bound: 0.0023462
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058029, 0.0057896
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106044, 0.0106076
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023696, upper bound: 0.0023696
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023714, upper bound: 0.0023700
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058015, 0.0057945
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106071, 0.0106087
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023998, upper bound: 0.0024062
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023996, upper bound: 0.0024062
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058084, 0.0057876
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106053, 0.0106106
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024184, upper bound: 0.0024239
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024220, upper bound: 0.0024242
time: 1.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024016, upper bound: 0.0024043
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024018, upper bound: 0.0024043
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024105, upper bound: 0.0024174
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024152, upper bound: 0.0024120
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023607, upper bound: 0.0023607
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023607, upper bound: 0.0023607
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023971, upper bound: 0.0023935
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023971, upper bound: 0.0023937
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024100, upper bound: 0.0024136
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024100, upper bound: 0.0024193
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024103, upper bound: 0.0024102
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024104, upper bound: 0.0024091
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024075, upper bound: 0.0024091
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024031, upper bound: 0.0024033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024166, upper bound: 0.0024231
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024166, upper bound: 0.0024232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024135, upper bound: 0.0024094
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024101, upper bound: 0.0024129
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024071, upper bound: 0.0024106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024071, upper bound: 0.0024072
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024135, upper bound: 0.0024094
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024101, upper bound: 0.0024129
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024071, upper bound: 0.0024106
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024108, upper bound: 0.0024072
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023489, upper bound: 0.0023462
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023489, upper bound: 0.0023462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023696, upper bound: 0.0023696
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023714, upper bound: 0.0023700
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023998, upper bound: 0.0024062
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0023996, upper bound: 0.0024062
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024184, upper bound: 0.0024239
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 1, lower bound: -0.0024220, upper bound: 0.0024242

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057787, 0.0057937
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106076, 0.0106036
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023763, upper bound: 0.0023796
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023760, upper bound: 0.0023796
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057796, 0.0057900
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106066, 0.0106039
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023950, upper bound: 0.0023977
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023950, upper bound: 0.0023977
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057878, 0.0058041
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106102, 0.0106059
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023879, upper bound: 0.0023951
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023879, upper bound: 0.0023951
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057901, 0.0058015
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106095, 0.0106065
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024020, upper bound: 0.0024047
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024082, upper bound: 0.0024035
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057766, 0.0057753
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106022, 0.0106026
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023880, upper bound: 0.0023821
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023820, upper bound: 0.0023847
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057752, 0.0057867
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106053, 0.0106022
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023859, upper bound: 0.0023862
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023896, upper bound: 0.0023859
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057728, 0.0057799
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106032, 0.0106013
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023978, upper bound: 0.0024049
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023978, upper bound: 0.0023995
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057682, 0.0057882
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106054, 0.0106001
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024099, upper bound: 0.0024191
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024099, upper bound: 0.0024192
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057896, 0.0057898
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106063, 0.0106062
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024014
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024014
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057903, 0.0057887
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106060, 0.0106064
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024007
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024007
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057734, 0.0057742
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106004, 0.0106002
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024090
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024090
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057753, 0.0057715
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105997, 0.0106007
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024035, upper bound: 0.0023945
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023941, upper bound: 0.0023943
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057761, 0.0057880
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106039, 0.0106007
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024136
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024066
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057762, 0.0057881
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106039, 0.0106007
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024129, upper bound: 0.0024202
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024131, upper bound: 0.0024188
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058044, 0.0057793
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106015, 0.0106080
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023989, upper bound: 0.0023987
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023991, upper bound: 0.0023989
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057981, 0.0057847
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106030, 0.0106063
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024066, upper bound: 0.0024106
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024074, upper bound: 0.0024094
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058110, 0.0057983
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106086, 0.0106116
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023961, upper bound: 0.0023988
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023961, upper bound: 0.0024014
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058170, 0.0057913
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106067, 0.0106132
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023979, upper bound: 0.0023979
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024017, upper bound: 0.0023979
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058022, 0.0057909
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106047, 0.0106074
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023989, upper bound: 0.0023989
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023991, upper bound: 0.0023989
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057967, 0.0057962
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106061, 0.0106059
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023463, upper bound: 0.0023488
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023463, upper bound: 0.0023488
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058099, 0.0058099
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106117, 0.0106113
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023982, upper bound: 0.0023988
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023961, upper bound: 0.0024014
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058156, 0.0058030
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106098, 0.0106128
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023963, upper bound: 0.0023985
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024014, upper bound: 0.0023946
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057941, 0.0057778
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106011, 0.0106050
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023471, upper bound: 0.0023446
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023475, upper bound: 0.0023441
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057912, 0.0057799
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106016, 0.0106043
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023553, upper bound: 0.0023605
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023553, upper bound: 0.0023553
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057825, 0.0057764
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106028, 0.0106041
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023988, upper bound: 0.0024058
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024061
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057835, 0.0057741
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106022, 0.0106044
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023994, upper bound: 0.0024058
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023995, upper bound: 0.0024061
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058084, 0.0057872
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106051, 0.0106105
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024126, upper bound: 0.0024143
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024127, upper bound: 0.0024143
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058084, 0.0057876
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106052, 0.0106105
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024109, upper bound: 0.0024153
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024109, upper bound: 0.0024160
time: 1.24 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023763, upper bound: 0.0023796
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023760, upper bound: 0.0023796
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023950, upper bound: 0.0023977
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023950, upper bound: 0.0023977
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023879, upper bound: 0.0023951
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023879, upper bound: 0.0023951
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024020, upper bound: 0.0024047
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024082, upper bound: 0.0024035
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023880, upper bound: 0.0023821
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023820, upper bound: 0.0023847
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023859, upper bound: 0.0023862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023896, upper bound: 0.0023859
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023978, upper bound: 0.0024049
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023978, upper bound: 0.0023995
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024099, upper bound: 0.0024191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024099, upper bound: 0.0024192
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024007
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024007
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024090
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024090
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024035, upper bound: 0.0023945
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023941, upper bound: 0.0023943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024136
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024028, upper bound: 0.0024066
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024129, upper bound: 0.0024202
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024131, upper bound: 0.0024188
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023989, upper bound: 0.0023987
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023991, upper bound: 0.0023989
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024066, upper bound: 0.0024106
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024074, upper bound: 0.0024094
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023961, upper bound: 0.0023988
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023961, upper bound: 0.0024014
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023979, upper bound: 0.0023979
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024017, upper bound: 0.0023979
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023989, upper bound: 0.0023989
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023991, upper bound: 0.0023989
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023463, upper bound: 0.0023488
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023463, upper bound: 0.0023488
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023982, upper bound: 0.0023988
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023961, upper bound: 0.0024014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023963, upper bound: 0.0023985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024014, upper bound: 0.0023946
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023471, upper bound: 0.0023446
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023475, upper bound: 0.0023441
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023553, upper bound: 0.0023605
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023553, upper bound: 0.0023553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023988, upper bound: 0.0024058
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024061
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023994, upper bound: 0.0024058
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0023995, upper bound: 0.0024061
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024126, upper bound: 0.0024143
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024127, upper bound: 0.0024143
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024109, upper bound: 0.0024153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 1, lower bound: -0.0024109, upper bound: 0.0024160

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057688, 0.0057821
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106045, 0.0106010
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023674, upper bound: 0.0023712
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023672, upper bound: 0.0023706
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057671, 0.0057937
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106076, 0.0106005
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023672, upper bound: 0.0023713
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023672, upper bound: 0.0023708
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057700, 0.0057811
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106043, 0.0106014
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023804, upper bound: 0.0023884
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023804, upper bound: 0.0023822
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057702, 0.0057804
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106042, 0.0106015
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023868, upper bound: 0.0023894
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023867, upper bound: 0.0023894
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057774, 0.0057925
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106071, 0.0106031
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023629, upper bound: 0.0023697
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023629, upper bound: 0.0023695
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057762, 0.0058041
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106102, 0.0106028
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023759, upper bound: 0.0023820
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023759, upper bound: 0.0023850
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057805, 0.0057918
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106070, 0.0106040
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023558, upper bound: 0.0023548
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023558, upper bound: 0.0023548
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057807, 0.0057919
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106071, 0.0106041
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023954
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023999, upper bound: 0.0023947
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057678, 0.0057636
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105989, 0.0106000
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023738, upper bound: 0.0023742
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023800, upper bound: 0.0023736
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057649, 0.0057649
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105992, 0.0105993
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023738, upper bound: 0.0023766
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023738, upper bound: 0.0023762
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057714, 0.0057834
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106044, 0.0106011
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023807, upper bound: 0.0023814
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023807, upper bound: 0.0023812
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057720, 0.0057827
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106042, 0.0106013
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023807, upper bound: 0.0023805
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023808
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057644, 0.0057740
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106017, 0.0105991
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023909, upper bound: 0.0023971
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023883, upper bound: 0.0023968
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057669, 0.0057711
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106009, 0.0105998
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023863, upper bound: 0.0023807
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023797, upper bound: 0.0023807
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057683, 0.0057883
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106054, 0.0106001
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024063, upper bound: 0.0024161
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024063, upper bound: 0.0024145
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057684, 0.0057883
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106054, 0.0106001
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023984
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023984
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057728, 0.0057735
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106018, 0.0106016
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023951
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023930, upper bound: 0.0024000
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057750, 0.0057730
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106017, 0.0106022
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023927
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023867
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057735, 0.0057728
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106016, 0.0106018
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023920
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023841
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057756, 0.0057720
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106014, 0.0106024
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023977, upper bound: 0.0023975
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023984, upper bound: 0.0023971
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057736, 0.0057744
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106003, 0.0106001
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023918, upper bound: 0.0023982
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023944, upper bound: 0.0023991
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057733, 0.0057743
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106003, 0.0106000
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023897
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023897
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057585, 0.0057547
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105951, 0.0105961
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023903, upper bound: 0.0023907
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023907
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057604, 0.0057547
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105951, 0.0105967
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023539, upper bound: 0.0023449
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023539, upper bound: 0.0023449
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057673, 0.0057818
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106023, 0.0105984
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023938, upper bound: 0.0024016
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023918, upper bound: 0.0024042
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057699, 0.0057797
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106017, 0.0105991
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023937, upper bound: 0.0023983
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023937, upper bound: 0.0023983
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057703, 0.0057844
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106024, 0.0105987
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024111
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024033
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057725, 0.0057823
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106019, 0.0105992
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024039, upper bound: 0.0024059
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024026, upper bound: 0.0024098
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057902, 0.0057646
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105976, 0.0106042
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023754, upper bound: 0.0023752
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023754, upper bound: 0.0023752
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057897, 0.0057632
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105972, 0.0106040
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023949
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023921
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057925, 0.0057818
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106016, 0.0106042
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023998
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023998
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057942, 0.0057791
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106009, 0.0106047
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023964, upper bound: 0.0024005
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023964, upper bound: 0.0024005
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058019, 0.0057866
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106053, 0.0106089
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023754
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023751, upper bound: 0.0023750
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057993, 0.0057884
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106057, 0.0106083
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023780
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023781
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058000, 0.0057774
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106028, 0.0106086
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023888, upper bound: 0.0023886
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023887, upper bound: 0.0023886
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058017, 0.0057746
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106021, 0.0106090
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023953, upper bound: 0.0023956
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023993, upper bound: 0.0023954
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057883, 0.0057762
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106007, 0.0106037
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023965
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023958
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057875, 0.0057748
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106004, 0.0106035
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023910, upper bound: 0.0023910
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023911, upper bound: 0.0023911
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058000, 0.0057981
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106084, 0.0106084
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023958, upper bound: 0.0023967
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023958, upper bound: 0.0023949
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057982, 0.0057999
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106088, 0.0106080
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023922, upper bound: 0.0023966
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023925, upper bound: 0.0024003
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058069, 0.0057968
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106082, 0.0106105
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023875, upper bound: 0.0023907
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023887, upper bound: 0.0023906
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0058094, 0.0057942
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106075, 0.0106112
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023830, upper bound: 0.0023832
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023900, upper bound: 0.0023829
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057825, 0.0057760
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106027, 0.0106040
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023889, upper bound: 0.0023956
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023899, upper bound: 0.0023955
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057827, 0.0057764
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106028, 0.0106041
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023954, upper bound: 0.0024030
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023967, upper bound: 0.0024023
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057836, 0.0057739
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106022, 0.0106043
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023895, upper bound: 0.0023958
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023895, upper bound: 0.0023966
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057839, 0.0057741
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106022, 0.0106044
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023485, upper bound: 0.0023534
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023485, upper bound: 0.0023534
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057914, 0.0057711
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106008, 0.0106060
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023936, upper bound: 0.0023967
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023936, upper bound: 0.0023967
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057939, 0.0057703
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106006, 0.0106066
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024010, upper bound: 0.0024066
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024054, upper bound: 0.0024058
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057924, 0.0057729
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106013, 0.0106062
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024035, upper bound: 0.0024024
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024008, upper bound: 0.0024069
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057938, 0.0057743
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106017, 0.0106066
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024096, upper bound: 0.0024127
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0024077, upper bound: 0.0024121
time: 1.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023674, upper bound: 0.0023712
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023672, upper bound: 0.0023706
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023672, upper bound: 0.0023713
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023672, upper bound: 0.0023708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023804, upper bound: 0.0023884
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023804, upper bound: 0.0023822
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023868, upper bound: 0.0023894
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023867, upper bound: 0.0023894
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023629, upper bound: 0.0023697
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023629, upper bound: 0.0023695
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023759, upper bound: 0.0023820
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023759, upper bound: 0.0023850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023558, upper bound: 0.0023548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023558, upper bound: 0.0023548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023954
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023999, upper bound: 0.0023947
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023738, upper bound: 0.0023742
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023800, upper bound: 0.0023736
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023738, upper bound: 0.0023766
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023738, upper bound: 0.0023762
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023807, upper bound: 0.0023814
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023807, upper bound: 0.0023812
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023807, upper bound: 0.0023805
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023808
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023909, upper bound: 0.0023971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023883, upper bound: 0.0023968
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023863, upper bound: 0.0023807
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023797, upper bound: 0.0023807
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024063, upper bound: 0.0024161
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024063, upper bound: 0.0024145
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023984
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023951
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023930, upper bound: 0.0024000
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023927
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023867
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023920
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023977, upper bound: 0.0023975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023984, upper bound: 0.0023971
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023918, upper bound: 0.0023982
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023944, upper bound: 0.0023991
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023897
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023897
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023903, upper bound: 0.0023907
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023907
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023539, upper bound: 0.0023449
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023539, upper bound: 0.0023449
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023938, upper bound: 0.0024016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023918, upper bound: 0.0024042
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023937, upper bound: 0.0023983
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023937, upper bound: 0.0023983
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024111
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024039, upper bound: 0.0024059
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024026, upper bound: 0.0024098
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023754, upper bound: 0.0023752
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023754, upper bound: 0.0023752
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023949
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023921
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023964, upper bound: 0.0024005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023964, upper bound: 0.0024005
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023754
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023751, upper bound: 0.0023750
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023780
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023781
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023888, upper bound: 0.0023886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023887, upper bound: 0.0023886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023953, upper bound: 0.0023956
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023993, upper bound: 0.0023954
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023958
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023910, upper bound: 0.0023910
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023911, upper bound: 0.0023911
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023958, upper bound: 0.0023967
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023958, upper bound: 0.0023949
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023922, upper bound: 0.0023966
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023925, upper bound: 0.0024003
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023875, upper bound: 0.0023907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023887, upper bound: 0.0023906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023830, upper bound: 0.0023832
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023900, upper bound: 0.0023829
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023889, upper bound: 0.0023956
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023899, upper bound: 0.0023955
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023954, upper bound: 0.0024030
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023967, upper bound: 0.0024023
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023895, upper bound: 0.0023958
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023895, upper bound: 0.0023966
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023485, upper bound: 0.0023534
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023485, upper bound: 0.0023534
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023936, upper bound: 0.0023967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0023936, upper bound: 0.0023967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024010, upper bound: 0.0024066
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024054, upper bound: 0.0024058
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024035, upper bound: 0.0024024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024008, upper bound: 0.0024069
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024096, upper bound: 0.0024127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 1, lower bound: -0.0024077, upper bound: 0.0024121

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057650, 0.0057790
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106035, 0.0105998
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023649, upper bound: 0.0023691
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023650, upper bound: 0.0023675
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057657, 0.0057782
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106033, 0.0105999
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023579, upper bound: 0.0023581
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023556, upper bound: 0.0023610
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057632, 0.0057902
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106066, 0.0105993
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0022996, upper bound: 0.0023036
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0022996, upper bound: 0.0023036
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057639, 0.0057895
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106064, 0.0105995
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023631, upper bound: 0.0023662
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023623, upper bound: 0.0023661
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057614, 0.0057749
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106028, 0.0105992
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023770, upper bound: 0.0023855
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023773, upper bound: 0.0023841
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057638, 0.0057717
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106019, 0.0105998
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023265, upper bound: 0.0023299
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023309, upper bound: 0.0023299
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057532, 0.0057648
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105998, 0.0105968
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023646, upper bound: 0.0023679
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023646, upper bound: 0.0023679
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057550, 0.0057634
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105995, 0.0105972
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023772, upper bound: 0.0023811
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023788, upper bound: 0.0023805
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057600, 0.0057759
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106029, 0.0105987
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023582, upper bound: 0.0023652
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023582, upper bound: 0.0023652
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057609, 0.0057724
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106020, 0.0105989
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023563, upper bound: 0.0023629
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023562, upper bound: 0.0023629
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057659, 0.0057923
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106069, 0.0105998
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023687, upper bound: 0.0023751
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023687, upper bound: 0.0023748
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057645, 0.0057940
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106074, 0.0105995
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023511, upper bound: 0.0023601
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023510, upper bound: 0.0023598
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057768, 0.0057887
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106060, 0.0106029
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023957, upper bound: 0.0023919
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023885, upper bound: 0.0023917
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057775, 0.0057883
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106059, 0.0106031
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023805, upper bound: 0.0023815
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023805, upper bound: 0.0023857
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057643, 0.0057604
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105980, 0.0105990
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023081, upper bound: 0.0023041
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023081, upper bound: 0.0023039
time: 1.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057646, 0.0057598
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105978, 0.0105991
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023735, upper bound: 0.0023737
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023799, upper bound: 0.0023737
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057614, 0.0057617
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105983, 0.0105982
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023710, upper bound: 0.0023741
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023710, upper bound: 0.0023741
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057617, 0.0057610
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105981, 0.0105983
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023603, upper bound: 0.0023667
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023603, upper bound: 0.0023617
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057618, 0.0057742
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106020, 0.0105986
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023782, upper bound: 0.0023791
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023782, upper bound: 0.0023786
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057628, 0.0057738
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106019, 0.0105989
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023140, upper bound: 0.0023107
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023140, upper bound: 0.0023107
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057624, 0.0057732
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106017, 0.0105988
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023806, upper bound: 0.0023806
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023806, upper bound: 0.0023806
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057639, 0.0057732
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106017, 0.0105992
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023150, upper bound: 0.0023098
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023150, upper bound: 0.0023098
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057607, 0.0057712
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106007, 0.0105979
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023674, upper bound: 0.0023752
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023674, upper bound: 0.0023753
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057616, 0.0057701
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106004, 0.0105981
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023797, upper bound: 0.0023886
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0023797, upper bound: 0.0023884
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057568, 0.0057595
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0105978, 0.0105971
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023186, upper bound: 0.0023156
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023186, upper bound: 0.0023155
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0000862, 0.0020094, 0.0000862, 0.0020094, -0.0019232, 0.0019232
1: 0.9913949, 0.9962445, 0.9913949, 0.9962445, -0.0048496, 0.0048496
2: -0.0096864, -0.0032848, -0.0096864, -0.0032848, -0.0057552, 0.0057711
3: 0.0029015, 0.0049129, 0.0029015, 0.0049129, -0.0020114, 0.0020114
4: 0.0011406, 0.0060726, 0.0011406, 0.0060726, -0.0049320, 0.0049320
5: 0.0039022, 0.0086155, 0.0039022, 0.0086155, -0.0047132, 0.0047132
6: -0.0024833, 0.0003110, -0.0024833, 0.0003110, -0.0027944, 0.0027944
7: -0.0098157, -0.0064219, -0.0098157, -0.0064219, -0.0033938, 0.0033938
8: 0.0002051, 0.0109784, 0.0002051, 0.0109784, -0.0106009, 0.0105967
9: -0.0066120, -0.0004573, -0.0066120, -0.0004573, -0.0061547, 0.0061547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023556, upper bound: 0.0023566
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0023556, upper bound: 0.0023566
time: 1.69 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023649, upper bound: 0.0023691
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023650, upper bound: 0.0023675
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023579, upper bound: 0.0023581
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023556, upper bound: 0.0023610
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0022996, upper bound: 0.0023036
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0022996, upper bound: 0.0023036
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023631, upper bound: 0.0023662
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023623, upper bound: 0.0023661
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023770, upper bound: 0.0023855
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023773, upper bound: 0.0023841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023265, upper bound: 0.0023299
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023309, upper bound: 0.0023299
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023646, upper bound: 0.0023679
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023646, upper bound: 0.0023679
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023772, upper bound: 0.0023811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023788, upper bound: 0.0023805
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023582, upper bound: 0.0023652
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023582, upper bound: 0.0023652
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023563, upper bound: 0.0023629
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023562, upper bound: 0.0023629
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023687, upper bound: 0.0023751
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023687, upper bound: 0.0023748
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023511, upper bound: 0.0023601
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023510, upper bound: 0.0023598
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023957, upper bound: 0.0023919
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023885, upper bound: 0.0023917
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023805, upper bound: 0.0023815
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023805, upper bound: 0.0023857
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023081, upper bound: 0.0023041
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023081, upper bound: 0.0023039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023735, upper bound: 0.0023737
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023799, upper bound: 0.0023737
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023710, upper bound: 0.0023741
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023710, upper bound: 0.0023741
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023603, upper bound: 0.0023667
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023603, upper bound: 0.0023617
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023782, upper bound: 0.0023791
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023782, upper bound: 0.0023786
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023140, upper bound: 0.0023107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023140, upper bound: 0.0023107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023806, upper bound: 0.0023806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023806, upper bound: 0.0023806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023150, upper bound: 0.0023098
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023150, upper bound: 0.0023098
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023674, upper bound: 0.0023752
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023674, upper bound: 0.0023753
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023797, upper bound: 0.0023886
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023797, upper bound: 0.0023884
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023186, upper bound: 0.0023156
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023186, upper bound: 0.0023155
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023556, upper bound: 0.0023566
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.61
Output dim: 1, lower bound: -0.0023556, upper bound: 0.0023566
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024063, upper bound: 0.0024161
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024063, upper bound: 0.0024145
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023984
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023951
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023930, upper bound: 0.0024000
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023927
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023867
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023920
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023838, upper bound: 0.0023841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023977, upper bound: 0.0023975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023984, upper bound: 0.0023971
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023918, upper bound: 0.0023982
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023944, upper bound: 0.0023991
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023897
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023847, upper bound: 0.0023897
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023903, upper bound: 0.0023907
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023938, upper bound: 0.0024016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023918, upper bound: 0.0024042
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023937, upper bound: 0.0023983
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023937, upper bound: 0.0023983
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024111
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023992, upper bound: 0.0024033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024039, upper bound: 0.0024059
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024026, upper bound: 0.0024098
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023754, upper bound: 0.0023752
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023754, upper bound: 0.0023752
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023949
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023921, upper bound: 0.0023921
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023964, upper bound: 0.0024005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023964, upper bound: 0.0024005
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023754
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023751, upper bound: 0.0023750
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023780
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023728, upper bound: 0.0023781
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023888, upper bound: 0.0023886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023887, upper bound: 0.0023886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023953, upper bound: 0.0023956
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023993, upper bound: 0.0023954
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023959, upper bound: 0.0023965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024002, upper bound: 0.0023958
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023910, upper bound: 0.0023910
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023911, upper bound: 0.0023911
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023958, upper bound: 0.0023967
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023958, upper bound: 0.0023949
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023922, upper bound: 0.0023966
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023925, upper bound: 0.0024003
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023875, upper bound: 0.0023907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023887, upper bound: 0.0023906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023830, upper bound: 0.0023832
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023900, upper bound: 0.0023829
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023889, upper bound: 0.0023956
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023899, upper bound: 0.0023955
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023954, upper bound: 0.0024030
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023967, upper bound: 0.0024023
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023895, upper bound: 0.0023958
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023895, upper bound: 0.0023966
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023936, upper bound: 0.0023967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0023936, upper bound: 0.0023967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024010, upper bound: 0.0024066
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024054, upper bound: 0.0024058
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024035, upper bound: 0.0024024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024008, upper bound: 0.0024069
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024096, upper bound: 0.0024127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 1, lower bound: -0.0024077, upper bound: 0.0024121

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.65 + 599.23 = 602.89 seconds
