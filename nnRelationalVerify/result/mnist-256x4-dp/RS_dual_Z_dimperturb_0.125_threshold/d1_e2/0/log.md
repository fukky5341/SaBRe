## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.3265e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000242, 0.0000242)
1: (0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000900, 0.0000900)
2: (-0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0005526, 0.0005526)
3: (0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000479, 0.0000479)
4: (0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004458, 0.0004458)
5: (0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001665, 0.0001665)
6: (-0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001913, 0.0001913)
7: (-0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001423, 0.0001423)
8: (0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006965, 0.0006965)
9: (-0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 1.26 = 2.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0000811, upper bound: 0.0000811

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000803, upper bound: 0.0000795
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000795, upper bound: 0.0000803
time: 0.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.02
Output dim: 1, lower bound: -0.0000803, upper bound: 0.0000795
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.02
Output dim: 1, lower bound: -0.0000795, upper bound: 0.0000803

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000242, 0.0000242
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000894, 0.0000892
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0005461, 0.0005443
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000474, 0.0000475
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004392, 0.0004407
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001665, 0.0001665
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001890, 0.0001884
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001406, 0.0001410
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006858, 0.0006882
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000691, upper bound: 0.0000687
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000690, upper bound: 0.0000687
time: 0.43 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000242, 0.0000242
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000892, 0.0000894
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0005443, 0.0005461
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000475, 0.0000474
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004407, 0.0004392
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001665, 0.0001665
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001884, 0.0001890
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001410, 0.0001406
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006882, 0.0006858
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000687, upper bound: 0.0000690
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000687, upper bound: 0.0000691
time: 0.45 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0000691, upper bound: 0.0000687
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0000690, upper bound: 0.0000687
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0000687, upper bound: 0.0000690
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0000687, upper bound: 0.0000691

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000239, 0.0000239
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000860, 0.0000855
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0005009, 0.0004955
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000446, 0.0000450
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004006, 0.0004049
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001598, 0.0001586
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001733, 0.0001714
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001314, 0.0001324
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006216, 0.0006287
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000242, 0.0000239
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000856, 0.0000892
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004973, 0.0005443
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000474, 0.0000447
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004392, 0.0004021
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001590, 0.0001665
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001721, 0.0001884
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001406, 0.0001318
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006858, 0.0006241
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000239, 0.0000239
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000857, 0.0000856
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004975, 0.0004973
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000447, 0.0000447
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004021, 0.0004023
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001591, 0.0001590
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001721, 0.0001721
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001318, 0.0001318
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006241, 0.0006243
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000242, 0.0000239
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000855, 0.0000894
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004955, 0.0005461
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000475, 0.0000446
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004407, 0.0004006
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001586, 0.0001665
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001714, 0.0001890
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001410, 0.0001314
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006882, 0.0006216
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654
time: 0.43 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000654, upper bound: 0.0000652
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 1, lower bound: -0.0000652, upper bound: 0.0000654

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000797, 0.0000793
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004460, 0.0004409
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000404, 0.0000407
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003635, 0.0003675
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001596, 0.0001585
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001539, 0.0001521
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001305, 0.0001315
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005346, 0.0005412
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000637
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000643, upper bound: 0.0000650
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000794, 0.0000794
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004428, 0.0004420
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000404, 0.0000405
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003643, 0.0003650
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001589, 0.0001587
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001528, 0.0001525
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001307, 0.0001309
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005360, 0.0005371
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000637
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000644, upper bound: 0.0000651
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000797, 0.0000793
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004460, 0.0004409
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000404, 0.0000407
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003635, 0.0003675
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001596, 0.0001585
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001539, 0.0001521
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001305, 0.0001315
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005346, 0.0005412
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000641
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000641, upper bound: 0.0000650
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000794, 0.0000794
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004428, 0.0004420
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000404, 0.0000405
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003643, 0.0003650
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001589, 0.0001587
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001528, 0.0001525
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001307, 0.0001309
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005360, 0.0005371
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000640
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000642, upper bound: 0.0000651
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000794, 0.0000794
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004420, 0.0004428
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000405, 0.0000404
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003650, 0.0003643
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001587, 0.0001589
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001525, 0.0001528
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001309, 0.0001307
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005371, 0.0005360
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000651, upper bound: 0.0000642
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000640, upper bound: 0.0000653
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000793, 0.0000797
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004409, 0.0004460
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000407, 0.0000404
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003675, 0.0003635
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001585, 0.0001596
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001521, 0.0001539
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001315, 0.0001305
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005412, 0.0005346
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000650, upper bound: 0.0000641
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000641, upper bound: 0.0000653
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000794, 0.0000794
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004420, 0.0004428
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000405, 0.0000404
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003650, 0.0003643
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001587, 0.0001589
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001525, 0.0001528
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001309, 0.0001307
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005371, 0.0005360
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000651, upper bound: 0.0000644
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000637, upper bound: 0.0000653
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000234, 0.0000234
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000793, 0.0000797
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0004409, 0.0004460
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000407, 0.0000404
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0003675, 0.0003635
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001585, 0.0001596
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001521, 0.0001539
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001315, 0.0001305
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0005412, 0.0005346
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 136

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000650, upper bound: 0.0000643
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0000637, upper bound: 0.0000653
time: 0.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000637
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000643, upper bound: 0.0000650
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000637
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000644, upper bound: 0.0000651
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000641
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000641, upper bound: 0.0000650
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000653, upper bound: 0.0000640
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000642, upper bound: 0.0000651
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000651, upper bound: 0.0000642
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000640, upper bound: 0.0000653
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000650, upper bound: 0.0000641
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000641, upper bound: 0.0000653
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000651, upper bound: 0.0000644
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000637, upper bound: 0.0000653
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000650, upper bound: 0.0000643
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 1, lower bound: -0.0000637, upper bound: 0.0000653

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0010721, 0.0010986, 0.0010721, 0.0010986, -0.0000241, 0.0000241
1: 0.9936401, 0.9937583, 0.9936401, 0.9937583, -0.0000890, 0.0000887
2: -0.0063458, -0.0055754, -0.0063458, -0.0055754, -0.0005424, 0.0005392
3: 0.0039326, 0.0039998, 0.0039326, 0.0039998, -0.0000470, 0.0000472
4: 0.0028235, 0.0034324, 0.0028235, 0.0034324, -0.0004354, 0.0004380
5: 0.0062247, 0.0063912, 0.0062247, 0.0063912, -0.0001665, 0.0001665
6: -0.0013238, -0.0010564, -0.0013238, -0.0010564, -0.0001877, 0.0001866
7: -0.0082173, -0.0080716, -0.0082173, -0.0080716, -0.0001403, 0.0001409
8: 0.0055771, 0.0065892, 0.0055771, 0.0065892, -0.0006792, 0.0006834
9: -0.0036820, -0.0036126, -0.0036820, -0.0036126, -0.0000694, 0.0000694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 80

### Candidate
type: RSZ, layer: 3, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 7
type: RSZ, layer: 7, pos: 225

Time for candidate selection: 2.00 seconds

### Candidate
type: RSZ, layer: 7, pos: 225

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

No RS candidates found

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.45 + 49.84 = 52.30 seconds
