## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0001407


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0007244, 0.0007244)
1: (-0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0002042, 0.0002042)
2: (-0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0015070, 0.0015070)
3: (0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001994, 0.0001994)
4: (0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0011262, 0.0011262)
5: (0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0003129, 0.0003129)
6: (0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002840, 0.0002840)
7: (-0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0010599, 0.0010599)
8: (-0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0008249, 0.0008249)
9: (-0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000712, 0.0000712)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.39 = 2.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0001773, upper bound: 0.0001772

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001649, upper bound: 0.0001667
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001667, upper bound: 0.0001649
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 5, lower bound: -0.0001649, upper bound: 0.0001667
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 5, lower bound: -0.0001667, upper bound: 0.0001649

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006071, 0.0006045
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001712, 0.0001704
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012628, 0.0012575
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001671, 0.0001664
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009398, 0.0009438
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002611, 0.0002622
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002370, 0.0002380
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008844, 0.0008882
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006913, 0.0006884
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000594, 0.0000596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001615, upper bound: 0.0001635
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001617, upper bound: 0.0001633
time: 0.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006045, 0.0006071
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001704, 0.0001712
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012575, 0.0012628
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001664, 0.0001671
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009438, 0.0009398
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002622, 0.0002611
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002380, 0.0002370
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008882, 0.0008844
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006884, 0.0006913
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000596, 0.0000594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001632, upper bound: 0.0001618
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001634, upper bound: 0.0001616
time: 0.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001615, upper bound: 0.0001635
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001617, upper bound: 0.0001633
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001632, upper bound: 0.0001618
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001634, upper bound: 0.0001616

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006013, 0.0005985
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001695, 0.0001687
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012508, 0.0012450
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001655, 0.0001648
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009304, 0.0009348
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002585, 0.0002597
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002346, 0.0002357
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008756, 0.0008797
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006847, 0.0006815
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000588, 0.0000591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001600, upper bound: 0.0001610
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001619
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006071, 0.0005987
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001712, 0.0001688
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012628, 0.0012455
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001671, 0.0001648
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009308, 0.0009438
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002586, 0.0002622
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002347, 0.0002380
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008760, 0.0008882
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006913, 0.0006818
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000588, 0.0000596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001602, upper bound: 0.0001609
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001617
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0005987, 0.0006000
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001688, 0.0001692
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012455, 0.0012481
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001648, 0.0001652
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009327, 0.0009308
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002591, 0.0002586
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002352, 0.0002347
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008778, 0.0008760
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006818, 0.0006832
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000589, 0.0000588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001593
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001608, upper bound: 0.0001601
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006045, 0.0006013
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001704, 0.0001695
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012575, 0.0012508
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001664, 0.0001655
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009348, 0.0009398
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002597, 0.0002611
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002357, 0.0002370
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008797, 0.0008844
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006884, 0.0006847
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000591, 0.0000594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001619, upper bound: 0.0001592
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001610, upper bound: 0.0001600
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001600, upper bound: 0.0001610
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001619
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001602, upper bound: 0.0001609
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001617
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001593
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001608, upper bound: 0.0001601
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001619, upper bound: 0.0001592
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 5, lower bound: -0.0001610, upper bound: 0.0001600

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0005987, 0.0005970
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001688, 0.0001683
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012454, 0.0012418
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001648, 0.0001643
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009281, 0.0009307
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002578, 0.0002586
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002340, 0.0002347
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008734, 0.0008759
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006817, 0.0006798
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000586, 0.0000588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0005995, 0.0005959
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001690, 0.0001680
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012471, 0.0012396
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001650, 0.0001640
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009264, 0.0009320
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002574, 0.0002589
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002336, 0.0002350
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008718, 0.0008771
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006827, 0.0006785
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000585, 0.0000589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006041, 0.0005973
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001703, 0.0001684
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012567, 0.0012424
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001663, 0.0001644
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009285, 0.0009392
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002580, 0.0002609
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002342, 0.0002368
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008738, 0.0008839
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006879, 0.0006801
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000587, 0.0000594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006050, 0.0005961
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001706, 0.0001681
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012584, 0.0012401
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001665, 0.0001641
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009268, 0.0009405
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002575, 0.0002613
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002337, 0.0002372
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008722, 0.0008851
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006889, 0.0006788
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000586, 0.0000594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0005961, 0.0005982
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001681, 0.0001686
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012401, 0.0012443
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001641, 0.0001647
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009299, 0.0009268
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002584, 0.0002575
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002345, 0.0002337
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008751, 0.0008722
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006788, 0.0006811
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000588, 0.0000586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0005973, 0.0005974
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001684, 0.0001684
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012424, 0.0012426
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001644, 0.0001644
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009287, 0.0009285
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002580, 0.0002580
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002342, 0.0002342
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008740, 0.0008738
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006801, 0.0006802
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000587, 0.0000587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006016, 0.0005995
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001696, 0.0001690
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012514, 0.0012471
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001656, 0.0001650
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009320, 0.0009352
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002589, 0.0002598
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002350, 0.0002358
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008771, 0.0008801
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006850, 0.0006827
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000589, 0.0000591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0084123, -0.0072625, -0.0084123, -0.0072625, -0.0006027, 0.0005987
1: -0.0053104, -0.0049862, -0.0053104, -0.0049862, -0.0001699, 0.0001688
2: -0.0006214, 0.0017703, -0.0006214, 0.0017703, -0.0012537, 0.0012454
3: 0.0015451, 0.0018616, 0.0015451, 0.0018616, -0.0001659, 0.0001648
4: 0.0047688, 0.0065562, 0.0047688, 0.0065562, -0.0009307, 0.0009370
5: 0.9968311, 0.9973277, 0.9968311, 0.9973277, -0.0002586, 0.0002603
6: 0.0050073, 0.0054581, 0.0050073, 0.0054581, -0.0002347, 0.0002363
7: -0.0046951, -0.0030130, -0.0046951, -0.0030130, -0.0008759, 0.0008818
8: -0.0068478, -0.0055386, -0.0068478, -0.0055386, -0.0006863, 0.0006817
9: -0.0035319, -0.0034189, -0.0035319, -0.0034189, -0.0000588, 0.0000592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
time: 0.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000524, upper bound: 0.0000532
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000532, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 5, lower bound: -0.0000528, upper bound: 0.0000528

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.73 + 34.83 = 37.56 seconds
