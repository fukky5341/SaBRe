## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.006503490000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142)
1: (0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415)
2: (-0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272)
3: (-0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832)
4: (0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747)
5: (-0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980)
6: (0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015)
7: (-0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0243378, 0.0243378)
8: (-0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058)
9: (-0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.81 + 3.54 = 4.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0069930, upper bound: 0.0069925

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069750, upper bound: 0.0069772
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069784, upper bound: 0.0069738
time: 2.15 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.63 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.63
Output dim: 6, lower bound: -0.0069750, upper bound: 0.0069772
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.63
Output dim: 6, lower bound: -0.0069784, upper bound: 0.0069738

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0240780, 0.0240898
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069323, upper bound: 0.0068752
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068714, upper bound: 0.0069343
time: 2.92 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0240898, 0.0240780
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069484, upper bound: 0.0069281
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069325, upper bound: 0.0069437
time: 2.89 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.67
Output dim: 6, lower bound: -0.0069323, upper bound: 0.0068752
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.67
Output dim: 6, lower bound: -0.0068714, upper bound: 0.0069343
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.67
Output dim: 6, lower bound: -0.0069484, upper bound: 0.0069281
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.67
Output dim: 6, lower bound: -0.0069325, upper bound: 0.0069437

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0235120, 0.0233979
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069082, upper bound: 0.0068179
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068567, upper bound: 0.0068489
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0233861, 0.0235199
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0065081
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0065081
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0238969, 0.0238503
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069241, upper bound: 0.0069064
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069275, upper bound: 0.0069034
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0238622, 0.0238893
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069051, upper bound: 0.0069208
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069097, upper bound: 0.0069170
time: 2.63 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.32 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0069082, upper bound: 0.0068179
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0068567, upper bound: 0.0068489
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0065081
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0065081
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0069241, upper bound: 0.0069064
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0069275, upper bound: 0.0069034
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0069051, upper bound: 0.0069208
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.32
Output dim: 6, lower bound: -0.0069097, upper bound: 0.0069170

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0231276, 0.0228885
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063385, upper bound: 0.0062850
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063385, upper bound: 0.0062850
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0230026, 0.0230074
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062671, upper bound: 0.0062590
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062671, upper bound: 0.0062590
time: 2.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0233738, 0.0235016
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062598, upper bound: 0.0062976
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062482, upper bound: 0.0063067
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0233861, 0.0235077
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064214, upper bound: 0.0064357
time: 4.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063954, upper bound: 0.0064794
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0235762, 0.0235337
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068089, upper bound: 0.0067721
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067930, upper bound: 0.0067890
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0235808, 0.0235296
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069231, upper bound: 0.0068512
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068672, upper bound: 0.0068987
time: 4.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0235904, 0.0236905
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063738, upper bound: 0.0063791
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063738, upper bound: 0.0063792
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0236661, 0.0236175
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069018, upper bound: 0.0069164
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0069090, upper bound: 0.0069122
time: 2.93 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0063385, upper bound: 0.0062850
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0063385, upper bound: 0.0062850
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0062671, upper bound: 0.0062590
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0062671, upper bound: 0.0062590
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0062598, upper bound: 0.0062976
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0062482, upper bound: 0.0063067
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0064214, upper bound: 0.0064357
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0063954, upper bound: 0.0064794
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0068089, upper bound: 0.0067721
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0067930, upper bound: 0.0067890
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0069231, upper bound: 0.0068512
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0068672, upper bound: 0.0068987
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0063738, upper bound: 0.0063791
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0063738, upper bound: 0.0063792
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0069018, upper bound: 0.0069164
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.33
Output dim: 6, lower bound: -0.0069090, upper bound: 0.0069122

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0232094, 0.0231180
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067867, upper bound: 0.0067139
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067385, upper bound: 0.0067498
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0231605, 0.0231608
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067480, upper bound: 0.0066875
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066824, upper bound: 0.0067412
time: 12.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0234210, 0.0232969
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067846, upper bound: 0.0067070
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067757, upper bound: 0.0067156
time: 3.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0233480, 0.0233719
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068021, upper bound: 0.0067825
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067498, upper bound: 0.0068345
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0232271, 0.0231732
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067911, upper bound: 0.0067419
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067249, upper bound: 0.0068042
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0232205, 0.0231785
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067876, upper bound: 0.0066994
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067030, upper bound: 0.0067950
time: 2.72 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 6.40 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067867, upper bound: 0.0067139
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067385, upper bound: 0.0067498
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067480, upper bound: 0.0066875
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0066824, upper bound: 0.0067412
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067846, upper bound: 0.0067070
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067757, upper bound: 0.0067156
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0068021, upper bound: 0.0067825
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067498, upper bound: 0.0068345
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067911, upper bound: 0.0067419
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067249, upper bound: 0.0068042
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067876, upper bound: 0.0066994
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.40
Output dim: 6, lower bound: -0.0067030, upper bound: 0.0067950

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0228251, 0.0226123
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063731, upper bound: 0.0063120
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063731, upper bound: 0.0063120
time: 2.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0227037, 0.0227299
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061943, upper bound: 0.0062099
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061943, upper bound: 0.0062091
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226470, 0.0225117
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067434, upper bound: 0.0066663
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067034, upper bound: 0.0066840
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0225115, 0.0226365
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065675, upper bound: 0.0065709
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065228, upper bound: 0.0066249
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0234065, 0.0232807
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067785, upper bound: 0.0067063
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067838, upper bound: 0.0066886
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0234048, 0.0232822
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066240, upper bound: 0.0065653
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066239, upper bound: 0.0065653
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0228218, 0.0227188
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066823, upper bound: 0.0066660
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066823, upper bound: 0.0066653
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226949, 0.0228443
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067452, upper bound: 0.0068257
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067455, upper bound: 0.0068298
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0228692, 0.0226515
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067679, upper bound: 0.0067220
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067682, upper bound: 0.0067160
time: 2.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0227055, 0.0228327
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066854, upper bound: 0.0067051
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066560, upper bound: 0.0067658
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0219594, 0.0216893
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064681, upper bound: 0.0063945
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064681, upper bound: 0.0063945
time: 2.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0217314, 0.0219227
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062515, upper bound: 0.0063342
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062515, upper bound: 0.0063342
time: 2.90 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.40 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0063731, upper bound: 0.0063120
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0063731, upper bound: 0.0063120
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0061943, upper bound: 0.0062099
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0061943, upper bound: 0.0062091
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067434, upper bound: 0.0066663
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067034, upper bound: 0.0066840
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0065675, upper bound: 0.0065709
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0065228, upper bound: 0.0066249
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067785, upper bound: 0.0067063
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067838, upper bound: 0.0066886
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0066240, upper bound: 0.0065653
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0066239, upper bound: 0.0065653
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0066823, upper bound: 0.0066660
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0066823, upper bound: 0.0066653
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067452, upper bound: 0.0068257
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067455, upper bound: 0.0068298
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067679, upper bound: 0.0067220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0067682, upper bound: 0.0067160
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0066854, upper bound: 0.0067051
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0066560, upper bound: 0.0067658
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0064681, upper bound: 0.0063945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0064681, upper bound: 0.0063945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0062515, upper bound: 0.0063342
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.40
Output dim: 6, lower bound: -0.0062515, upper bound: 0.0063342

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226473, 0.0225118
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066272, upper bound: 0.0065023
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065783, upper bound: 0.0065506
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226470, 0.0225120
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065678, upper bound: 0.0065471
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065664, upper bound: 0.0065481
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0223798, 0.0224161
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065113, upper bound: 0.0065215
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065113, upper bound: 0.0065642
time: 2.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0222911, 0.0225058
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061672, upper bound: 0.0062207
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061672, upper bound: 0.0062208
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0229691, 0.0228477
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062516, upper bound: 0.0061917
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062516, upper bound: 0.0061917
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0229675, 0.0228433
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067520, upper bound: 0.0066217
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066971, upper bound: 0.0066568
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0233714, 0.0232347
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064894, upper bound: 0.0063593
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064115, upper bound: 0.0064305
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0233572, 0.0232494
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065362, upper bound: 0.0064872
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065391, upper bound: 0.0065207
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0227918, 0.0226724
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060409, upper bound: 0.0060405
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060409, upper bound: 0.0060405
time: 2.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0227755, 0.0226867
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065837, upper bound: 0.0065507
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065673, upper bound: 0.0065625
time: 2.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0225035, 0.0226115
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063263, upper bound: 0.0064021
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063263, upper bound: 0.0064016
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0224726, 0.0226529
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066354, upper bound: 0.0066716
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065996, upper bound: 0.0067182
time: 2.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226086, 0.0224055
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067633, upper bound: 0.0066973
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067290, upper bound: 0.0067172
time: 2.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226109, 0.0223909
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066970, upper bound: 0.0065747
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066405, upper bound: 0.0066490
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0222182, 0.0222350
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065745, upper bound: 0.0065759
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065569, upper bound: 0.0065967
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0221078, 0.0223617
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066260, upper bound: 0.0066745
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065971, upper bound: 0.0067366
time: 2.93 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 6.66 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0066272, upper bound: 0.0065023
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065783, upper bound: 0.0065506
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065678, upper bound: 0.0065471
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065664, upper bound: 0.0065481
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065113, upper bound: 0.0065215
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065113, upper bound: 0.0065642
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0061672, upper bound: 0.0062207
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0061672, upper bound: 0.0062208
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0062516, upper bound: 0.0061917
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0062516, upper bound: 0.0061917
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0067520, upper bound: 0.0066217
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0066971, upper bound: 0.0066568
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0064894, upper bound: 0.0063593
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0064115, upper bound: 0.0064305
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065362, upper bound: 0.0064872
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065391, upper bound: 0.0065207
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0060409, upper bound: 0.0060405
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0060409, upper bound: 0.0060405
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065837, upper bound: 0.0065507
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065673, upper bound: 0.0065625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0063263, upper bound: 0.0064021
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0063263, upper bound: 0.0064016
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0066354, upper bound: 0.0066716
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065996, upper bound: 0.0067182
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0067633, upper bound: 0.0066973
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0067290, upper bound: 0.0067172
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0066970, upper bound: 0.0065747
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0066405, upper bound: 0.0066490
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065745, upper bound: 0.0065759
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065569, upper bound: 0.0065967
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0066260, upper bound: 0.0066745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.66
Output dim: 6, lower bound: -0.0065971, upper bound: 0.0067366

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0225143, 0.0222914
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060099, upper bound: 0.0059446
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060099, upper bound: 0.0059446
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0224269, 0.0223926
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065478, upper bound: 0.0064997
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064756, upper bound: 0.0065182
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226345, 0.0224966
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065612, upper bound: 0.0065392
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065588, upper bound: 0.0065399
time: 2.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226317, 0.0224986
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065590, upper bound: 0.0065410
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065556, upper bound: 0.0065416
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0222224, 0.0221837
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064423, upper bound: 0.0064027
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064424, upper bound: 0.0064035
time: 2.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0221473, 0.0222584
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060204, upper bound: 0.0060876
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060204, upper bound: 0.0060883
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0222770, 0.0219412
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061796, upper bound: 0.0060916
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061796, upper bound: 0.0060916
time: 2.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0220654, 0.0221740
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066336, upper bound: 0.0065861
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066306, upper bound: 0.0066337
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0228329, 0.0225985
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064606, upper bound: 0.0063267
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064204, upper bound: 0.0063615
time: 2.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0227064, 0.0227290
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065235, upper bound: 0.0064984
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065132, upper bound: 0.0065060
time: 2.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0224118, 0.0222747
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064361, upper bound: 0.0064092
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063886, upper bound: 0.0064084
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0223635, 0.0223253
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065431, upper bound: 0.0065033
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064818, upper bound: 0.0065367
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0223268, 0.0224222
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060583, upper bound: 0.0061043
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060583, upper bound: 0.0061043
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0222420, 0.0225104
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064573, upper bound: 0.0065653
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064564, upper bound: 0.0065702
time: 2.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226090, 0.0224058
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061784, upper bound: 0.0061294
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061530, upper bound: 0.0061294
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0226088, 0.0224060
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061522, upper bound: 0.0061410
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061530, upper bound: 0.0061406
time: 2.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0219366, 0.0216135
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065976, upper bound: 0.0064415
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065458, upper bound: 0.0064744
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0218335, 0.0217402
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065784, upper bound: 0.0065693
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065783, upper bound: 0.0066199
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0216819, 0.0216624
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064348, upper bound: 0.0064362
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064078, upper bound: 0.0064370
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0216456, 0.0216875
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065473, upper bound: 0.0065452
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064930, upper bound: 0.0065882
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0214296, 0.0214599
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064567, upper bound: 0.0065344
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064826, upper bound: 0.0065332
time: 4.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0212061, 0.0216864
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064497, upper bound: 0.0065744
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064488, upper bound: 0.0065790
time: 2.83 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 7.34 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0060099, upper bound: 0.0059446
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0060099, upper bound: 0.0059446
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065478, upper bound: 0.0064997
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064756, upper bound: 0.0065182
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065612, upper bound: 0.0065392
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065588, upper bound: 0.0065399
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065590, upper bound: 0.0065410
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065556, upper bound: 0.0065416
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064423, upper bound: 0.0064027
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064424, upper bound: 0.0064035
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0060204, upper bound: 0.0060876
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0060204, upper bound: 0.0060883
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0061796, upper bound: 0.0060916
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0061796, upper bound: 0.0060916
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0066336, upper bound: 0.0065861
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0066306, upper bound: 0.0066337
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064606, upper bound: 0.0063267
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064204, upper bound: 0.0063615
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065235, upper bound: 0.0064984
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065132, upper bound: 0.0065060
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064361, upper bound: 0.0064092
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0063886, upper bound: 0.0064084
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065431, upper bound: 0.0065033
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064818, upper bound: 0.0065367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0060583, upper bound: 0.0061043
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0060583, upper bound: 0.0061043
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064573, upper bound: 0.0065653
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064564, upper bound: 0.0065702
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0061784, upper bound: 0.0061294
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0061530, upper bound: 0.0061294
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0061522, upper bound: 0.0061410
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0061530, upper bound: 0.0061406
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065976, upper bound: 0.0064415
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065458, upper bound: 0.0064744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065784, upper bound: 0.0065693
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065783, upper bound: 0.0066199
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064348, upper bound: 0.0064362
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064078, upper bound: 0.0064370
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0065473, upper bound: 0.0065452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064930, upper bound: 0.0065882
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064567, upper bound: 0.0065344
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064826, upper bound: 0.0065332
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064497, upper bound: 0.0065744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.34
Output dim: 6, lower bound: -0.0064488, upper bound: 0.0065790

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0217616, 0.0214985
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064027, upper bound: 0.0063519
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063975, upper bound: 0.0063530
time: 3.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0215327, 0.0216791
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064518, upper bound: 0.0064639
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064118, upper bound: 0.0064926
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0224641, 0.0222922
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065146, upper bound: 0.0064574
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064731, upper bound: 0.0064935
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0224251, 0.0223263
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065448, upper bound: 0.0065169
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065329, upper bound: 0.0065258
time: 2.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0224613, 0.0222941
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064260, upper bound: 0.0063656
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063925, upper bound: 0.0064124
time: 2.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0224223, 0.0223283
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064280, upper bound: 0.0063468
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063856, upper bound: 0.0064175
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0215738, 0.0215545
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065362, upper bound: 0.0064136
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064832, upper bound: 0.0064513
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0214460, 0.0216650
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066267, upper bound: 0.0066122
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066175, upper bound: 0.0066297
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0227128, 0.0227316
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064496, upper bound: 0.0064978
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065221, upper bound: 0.0064817
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0227098, 0.0227355
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063739, upper bound: 0.0063111
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063143, upper bound: 0.0063673
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0220030, 0.0218349
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065136, upper bound: 0.0064461
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064658, upper bound: 0.0064736
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0218731, 0.0219499
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064774, upper bound: 0.0065148
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064525, upper bound: 0.0065320
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0222279, 0.0224950
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063388, upper bound: 0.0064409
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063372, upper bound: 0.0064552
time: 2.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0222266, 0.0224979
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0058386, upper bound: 0.0058936
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0058386, upper bound: 0.0058936
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0217937, 0.0213719
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065837, upper bound: 0.0064155
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065739, upper bound: 0.0064282
time: 3.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0216951, 0.0214533
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064245, upper bound: 0.0062624
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063250, upper bound: 0.0063515
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0211766, 0.0208662
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064891, upper bound: 0.0063670
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064187, upper bound: 0.0064508
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0209595, 0.0210936
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065736, upper bound: 0.0066044
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065721, upper bound: 0.0066148
time: 2.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0214164, 0.0213851
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065182, upper bound: 0.0064833
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064748, upper bound: 0.0065162
time: 2.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0213432, 0.0214458
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059279, upper bound: 0.0060004
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059279, upper bound: 0.0060009
time: 2.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0213943, 0.0214101
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064746, upper bound: 0.0065207
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064681, upper bound: 0.0065222
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142
1: 0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415
2: -0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272
3: -0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832
4: 0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747
5: -0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980
6: 0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015
7: -0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0213798, 0.0214166
8: -0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058
9: -0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064785, upper bound: 0.0065098
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064713, upper bound: 0.0065294
time: 3.13 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 6.47 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064027, upper bound: 0.0063519
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063975, upper bound: 0.0063530
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064518, upper bound: 0.0064639
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064118, upper bound: 0.0064926
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065146, upper bound: 0.0064574
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064731, upper bound: 0.0064935
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065448, upper bound: 0.0065169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065329, upper bound: 0.0065258
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064260, upper bound: 0.0063656
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063925, upper bound: 0.0064124
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064280, upper bound: 0.0063468
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063856, upper bound: 0.0064175
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065362, upper bound: 0.0064136
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064832, upper bound: 0.0064513
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0066267, upper bound: 0.0066122
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0066175, upper bound: 0.0066297
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064496, upper bound: 0.0064978
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065221, upper bound: 0.0064817
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063739, upper bound: 0.0063111
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063143, upper bound: 0.0063673
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065136, upper bound: 0.0064461
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064658, upper bound: 0.0064736
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064774, upper bound: 0.0065148
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064525, upper bound: 0.0065320
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063388, upper bound: 0.0064409
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063372, upper bound: 0.0064552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0058386, upper bound: 0.0058936
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0058386, upper bound: 0.0058936
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065837, upper bound: 0.0064155
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065739, upper bound: 0.0064282
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064245, upper bound: 0.0062624
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0063250, upper bound: 0.0063515
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064891, upper bound: 0.0063670
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064187, upper bound: 0.0064508
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065736, upper bound: 0.0066044
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065721, upper bound: 0.0066148
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0065182, upper bound: 0.0064833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064748, upper bound: 0.0065162
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0059279, upper bound: 0.0060004
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0059279, upper bound: 0.0060009
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064746, upper bound: 0.0065207
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064681, upper bound: 0.0065222
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064785, upper bound: 0.0065098
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.47
Output dim: 6, lower bound: -0.0064713, upper bound: 0.0065294
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.47
Output dim: 6, lower bound: -0.0064497, upper bound: 0.0065744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.47
Output dim: 6, lower bound: -0.0064488, upper bound: 0.0065790

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.36 + 597.22 = 601.58 seconds
