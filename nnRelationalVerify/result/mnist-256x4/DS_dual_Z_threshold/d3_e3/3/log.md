## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001979395


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078547, 0.0078547)
1: (-0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022145, 0.0022145)
2: (-0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163393, 0.0163393)
3: (-0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021622, 0.0021622)
4: (0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0122110, 0.0122110)
5: (0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033926, 0.0033926)
6: (0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030794, 0.0030794)
7: (-0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114919, 0.0114919)
8: (-0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089442, 0.0089442)
9: (-0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007717, 0.0007717)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.23 + 2.78 = 5.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0023286, upper bound: 0.0023287

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0023167, upper bound: 0.0022871
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022872, upper bound: 0.0023167
time: 2.04 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.24
Output dim: 5, lower bound: -0.0023167, upper bound: 0.0022871
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.24
Output dim: 5, lower bound: -0.0022872, upper bound: 0.0023167

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078385, 0.0078436
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022100, 0.0022114
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163056, 0.0163163
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021578, 0.0021592
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121938, 0.0121858
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033878, 0.0033856
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030751, 0.0030731
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114757, 0.0114682
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089257, 0.0089315
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007706, 0.0007701

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022091, upper bound: 0.0021793
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022091, upper bound: 0.0021793
time: 1.54 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078436, 0.0078385
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022114, 0.0022100
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163163, 0.0163056
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021592, 0.0021578
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121858, 0.0121938
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033856, 0.0033878
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030731, 0.0030751
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114682, 0.0114757
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089315, 0.0089257
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007701, 0.0007706

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091
time: 1.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.09 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.09
Output dim: 5, lower bound: -0.0022091, upper bound: 0.0021793
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.09
Output dim: 5, lower bound: -0.0022091, upper bound: 0.0021793
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.09
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.09
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078105, 0.0078355
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022021, 0.0022091
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162475, 0.0162995
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021501, 0.0021570
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121812, 0.0121424
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033843, 0.0033735
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030719, 0.0030621
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114639, 0.0114273
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088939, 0.0089224
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007698, 0.0007673

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021949, upper bound: 0.0021624
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021944, upper bound: 0.0021653
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078385, 0.0078157
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022100, 0.0022035
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163056, 0.0162581
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021578, 0.0021515
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121503, 0.0121858
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033757, 0.0033856
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030641, 0.0030731
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114348, 0.0114682
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089257, 0.0088997
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007678, 0.0007701

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021949, upper bound: 0.0021624
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021944, upper bound: 0.0021654
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078157, 0.0078305
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022035, 0.0022077
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162581, 0.0162890
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021515, 0.0021556
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121734, 0.0121503
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033821, 0.0033757
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030699, 0.0030641
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114565, 0.0114348
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088997, 0.0089166
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007693, 0.0007678

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021654, upper bound: 0.0021944
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021624, upper bound: 0.0021949
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078436, 0.0078105
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022114, 0.0022021
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163163, 0.0162475
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021592, 0.0021501
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121424, 0.0121938
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033735, 0.0033878
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030621, 0.0030751
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114273, 0.0114757
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089315, 0.0088939
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007673, 0.0007706

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021654, upper bound: 0.0021944
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021624, upper bound: 0.0021949
time: 1.68 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.59 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021949, upper bound: 0.0021624
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021944, upper bound: 0.0021653
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021949, upper bound: 0.0021624
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021944, upper bound: 0.0021654
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021654, upper bound: 0.0021944
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021624, upper bound: 0.0021949
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021654, upper bound: 0.0021944
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.59
Output dim: 5, lower bound: -0.0021624, upper bound: 0.0021949

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077834, 0.0078082
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021944, 0.0022014
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161910, 0.0162426
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021426, 0.0021495
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121387, 0.0121002
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033725, 0.0033618
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030612, 0.0030515
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114239, 0.0113876
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088630, 0.0088912
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007671, 0.0007647

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021790, upper bound: 0.0021276
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021618, upper bound: 0.0021464
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077829, 0.0078084
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021943, 0.0022015
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161899, 0.0162430
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021425, 0.0021495
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121390, 0.0120993
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033726, 0.0033616
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030613, 0.0030513
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114242, 0.0113868
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088624, 0.0088914
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007671, 0.0007646

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021784, upper bound: 0.0021325
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021602, upper bound: 0.0021491
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078115, 0.0077878
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022024, 0.0021957
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162495, 0.0162003
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021504, 0.0021439
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121071, 0.0121439
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033637, 0.0033739
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030532, 0.0030625
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113941, 0.0114288
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088950, 0.0088681
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007651, 0.0007674

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021790, upper bound: 0.0021276
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021618, upper bound: 0.0021463
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078110, 0.0077885
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022022, 0.0021959
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162484, 0.0162017
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021502, 0.0021440
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121081, 0.0121431
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033640, 0.0033737
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030535, 0.0030623
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113951, 0.0114280
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088944, 0.0088688
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007652, 0.0007674

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021784, upper bound: 0.0021324
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021602, upper bound: 0.0021492
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077885, 0.0078031
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021959, 0.0022000
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162017, 0.0162320
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021440, 0.0021481
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121308, 0.0121081
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033703, 0.0033640
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030592, 0.0030535
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114164, 0.0113951
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088688, 0.0088854
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007666, 0.0007652

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021492, upper bound: 0.0021602
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021324, upper bound: 0.0021783
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077878, 0.0078033
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021957, 0.0022000
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162003, 0.0162325
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021439, 0.0021481
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121311, 0.0121071
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033704, 0.0033637
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030593, 0.0030532
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0114167, 0.0113941
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088681, 0.0088857
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007666, 0.0007651

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021464, upper bound: 0.0021618
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021277, upper bound: 0.0021790
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078166, 0.0077829
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022038, 0.0021943
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162602, 0.0161899
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021518, 0.0021425
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120993, 0.0121518
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033616, 0.0033761
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030513, 0.0030645
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113868, 0.0114362
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089008, 0.0088624
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007646, 0.0007679

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021492, upper bound: 0.0021602
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021324, upper bound: 0.0021784
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078160, 0.0077834
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022036, 0.0021944
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162588, 0.0161910
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021516, 0.0021426
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121002, 0.0121508
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033618, 0.0033759
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030515, 0.0030643
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113876, 0.0114353
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089001, 0.0088630
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007647, 0.0007679

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021464, upper bound: 0.0021617
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021277, upper bound: 0.0021790
time: 1.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021790, upper bound: 0.0021276
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021618, upper bound: 0.0021464
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021784, upper bound: 0.0021325
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021602, upper bound: 0.0021491
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021790, upper bound: 0.0021276
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021618, upper bound: 0.0021463
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021784, upper bound: 0.0021324
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021602, upper bound: 0.0021492
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021492, upper bound: 0.0021602
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021324, upper bound: 0.0021783
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021464, upper bound: 0.0021618
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021277, upper bound: 0.0021790
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021492, upper bound: 0.0021602
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021324, upper bound: 0.0021784
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021464, upper bound: 0.0021617
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 5, lower bound: -0.0021277, upper bound: 0.0021790

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076439, 0.0077112
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021551, 0.0021741
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159008, 0.0160410
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021042, 0.0021228
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119880, 0.0118833
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033306, 0.0033015
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030232, 0.0029968
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112821, 0.0111835
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087041, 0.0087808
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007576, 0.0007510

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021772, upper bound: 0.0020871
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021323, upper bound: 0.0021258
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076856, 0.0076687
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021669, 0.0021621
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159876, 0.0159525
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021157, 0.0021111
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119219, 0.0119481
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033122, 0.0033195
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030065, 0.0030131
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112198, 0.0112445
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087516, 0.0087324
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007534, 0.0007550

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021600, upper bound: 0.0021053
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021177, upper bound: 0.0021446
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076434, 0.0077110
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021549, 0.0021740
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158997, 0.0160405
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021041, 0.0021227
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119877, 0.0118825
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033305, 0.0033013
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030231, 0.0029966
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112817, 0.0111827
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087035, 0.0087806
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007575, 0.0007509

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021766, upper bound: 0.0020915
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021317, upper bound: 0.0021307
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076857, 0.0076689
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021669, 0.0021621
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159878, 0.0159528
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021157, 0.0021111
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119221, 0.0119483
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033123, 0.0033196
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030066, 0.0030132
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112201, 0.0112447
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087518, 0.0087326
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007534, 0.0007551

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021584, upper bound: 0.0021089
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021151, upper bound: 0.0021473
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076710, 0.0076904
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021627, 0.0021682
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159572, 0.0159975
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021117, 0.0021170
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119555, 0.0119254
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033216, 0.0033132
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030150, 0.0030074
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112515, 0.0112231
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087350, 0.0087571
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007555, 0.0007536

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021772, upper bound: 0.0020871
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021323, upper bound: 0.0021258
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077126, 0.0076483
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021745, 0.0021564
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160439, 0.0159101
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021232, 0.0021055
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118902, 0.0119902
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033035, 0.0033312
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029985, 0.0030238
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111900, 0.0112841
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087824, 0.0087092
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007514, 0.0007577

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021600, upper bound: 0.0021052
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021177, upper bound: 0.0021445
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076704, 0.0076903
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021626, 0.0021682
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159561, 0.0159975
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021115, 0.0021170
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119555, 0.0119246
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033216, 0.0033130
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030150, 0.0030072
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112515, 0.0112223
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087344, 0.0087570
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007555, 0.0007536

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021766, upper bound: 0.0020915
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021317, upper bound: 0.0021307
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077128, 0.0076490
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021745, 0.0021565
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160442, 0.0159115
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021232, 0.0021056
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118912, 0.0119904
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033037, 0.0033313
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029988, 0.0030238
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111910, 0.0112843
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087826, 0.0087100
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007515, 0.0007577

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021584, upper bound: 0.0021089
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021151, upper bound: 0.0021474
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076490, 0.0077065
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021565, 0.0021728
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159115, 0.0160312
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021056, 0.0021215
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119807, 0.0118912
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033286, 0.0033037
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030214, 0.0029988
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112752, 0.0111910
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087100, 0.0087755
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007571, 0.0007515

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021474, upper bound: 0.0021151
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021090, upper bound: 0.0021584
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076903, 0.0076636
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021682, 0.0021607
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159975, 0.0159419
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021170, 0.0021097
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119139, 0.0119555
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033100, 0.0033216
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030045, 0.0030150
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112123, 0.0112515
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087570, 0.0087266
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007529, 0.0007555

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021307, upper bound: 0.0021317
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020915, upper bound: 0.0021766
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076483, 0.0077063
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021564, 0.0021727
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159101, 0.0160308
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021055, 0.0021214
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119804, 0.0118902
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033285, 0.0033035
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030213, 0.0029985
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112749, 0.0111900
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087092, 0.0087753
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007571, 0.0007514

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021446, upper bound: 0.0021177
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021053, upper bound: 0.0021600
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076904, 0.0076638
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021682, 0.0021607
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159975, 0.0159423
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021170, 0.0021097
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119143, 0.0119555
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033101, 0.0033216
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030046, 0.0030150
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112127, 0.0112515
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087571, 0.0087268
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007529, 0.0007555

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021258, upper bound: 0.0021324
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020871, upper bound: 0.0021772
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076761, 0.0076857
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021642, 0.0021669
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159678, 0.0159878
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021131, 0.0021157
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119483, 0.0119333
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033196, 0.0033154
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030132, 0.0030094
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112447, 0.0112306
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087408, 0.0087518
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007551, 0.0007541

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021474, upper bound: 0.0021151
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021090, upper bound: 0.0021585
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077174, 0.0076434
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021758, 0.0021549
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160538, 0.0158997
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021245, 0.0021041
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118825, 0.0119976
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033013, 0.0033333
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029966, 0.0030256
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111827, 0.0112911
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087879, 0.0087035
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007509, 0.0007582

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021307, upper bound: 0.0021317
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020871, upper bound: 0.0021766
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076754, 0.0076856
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021640, 0.0021669
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159664, 0.0159876
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021129, 0.0021157
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119481, 0.0119323
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033195, 0.0033152
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030131, 0.0030092
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112445, 0.0112296
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087400, 0.0087516
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007550, 0.0007540

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021446, upper bound: 0.0021178
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021053, upper bound: 0.0021600
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077174, 0.0076439
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021758, 0.0021551
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160538, 0.0159008
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021245, 0.0021042
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118833, 0.0119976
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033015, 0.0033333
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029968, 0.0030256
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111835, 0.0112911
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087879, 0.0087041
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007510, 0.0007582

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021258, upper bound: 0.0021323
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020871, upper bound: 0.0021773
time: 1.84 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 6.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021772, upper bound: 0.0020871
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021323, upper bound: 0.0021258
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021600, upper bound: 0.0021053
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021177, upper bound: 0.0021446
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021766, upper bound: 0.0020915
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021317, upper bound: 0.0021307
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021584, upper bound: 0.0021089
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021151, upper bound: 0.0021473
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021772, upper bound: 0.0020871
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021323, upper bound: 0.0021258
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021600, upper bound: 0.0021052
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021177, upper bound: 0.0021445
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021766, upper bound: 0.0020915
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021317, upper bound: 0.0021307
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021584, upper bound: 0.0021089
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021151, upper bound: 0.0021474
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021474, upper bound: 0.0021151
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021090, upper bound: 0.0021584
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021307, upper bound: 0.0021317
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0020915, upper bound: 0.0021766
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021446, upper bound: 0.0021177
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021053, upper bound: 0.0021600
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021258, upper bound: 0.0021324
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0020871, upper bound: 0.0021772
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021474, upper bound: 0.0021151
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021090, upper bound: 0.0021585
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021307, upper bound: 0.0021317
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0020871, upper bound: 0.0021766
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021446, upper bound: 0.0021178
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021053, upper bound: 0.0021600
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0021258, upper bound: 0.0021323
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.05
Output dim: 5, lower bound: -0.0020871, upper bound: 0.0021773

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076468, 0.0077367
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021559, 0.0021813
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159070, 0.0160939
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021050, 0.0021298
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120276, 0.0118879
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033416, 0.0033028
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030332, 0.0029979
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113193, 0.0111878
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087075, 0.0088098
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007601, 0.0007512

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020351, upper bound: 0.0019644
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020326, upper bound: 0.0019648
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076682, 0.0077142
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021620, 0.0021749
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159514, 0.0160471
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021109, 0.0021236
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119926, 0.0119211
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033319, 0.0033120
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030244, 0.0030063
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112863, 0.0112191
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087318, 0.0087842
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007579, 0.0007533

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020003, upper bound: 0.0019942
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019956, upper bound: 0.0019958
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076885, 0.0076947
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021677, 0.0021694
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159937, 0.0160065
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021165, 0.0021182
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119622, 0.0119527
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033235, 0.0033208
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030167, 0.0030143
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112578, 0.0112488
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087550, 0.0087620
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007559, 0.0007553

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020225, upper bound: 0.0019778
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020210, upper bound: 0.0019806
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077106, 0.0076716
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021739, 0.0021629
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160397, 0.0159586
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021226, 0.0021119
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119264, 0.0119871
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033135, 0.0033304
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030077, 0.0030230
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112241, 0.0112812
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087802, 0.0087357
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007537, 0.0007575

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019875, upper bound: 0.0020091
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0020100
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076463, 0.0077365
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021558, 0.0021812
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159058, 0.0160934
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021049, 0.0021297
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120272, 0.0118870
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033415, 0.0033026
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030331, 0.0029977
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113190, 0.0111870
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087069, 0.0088096
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007600, 0.0007512

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020333, upper bound: 0.0019698
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020307, upper bound: 0.0019706
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076682, 0.0077140
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021620, 0.0021749
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159514, 0.0160466
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021109, 0.0021235
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119922, 0.0119211
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033318, 0.0033120
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030243, 0.0030063
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112860, 0.0112191
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087318, 0.0087839
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007578, 0.0007533

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019978, upper bound: 0.0019982
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019931, upper bound: 0.0019995
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076886, 0.0076946
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021677, 0.0021694
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159939, 0.0160064
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021165, 0.0021182
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119622, 0.0119529
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033234, 0.0033209
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030167, 0.0030143
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112577, 0.0112490
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087551, 0.0087619
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007559, 0.0007553

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020205, upper bound: 0.0019818
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020188, upper bound: 0.0019851
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077109, 0.0076718
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021740, 0.0021630
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160403, 0.0159589
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021227, 0.0021119
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119267, 0.0119875
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033136, 0.0033305
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030077, 0.0030231
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112244, 0.0112816
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087805, 0.0087359
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007537, 0.0007575

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019848, upper bound: 0.0020121
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0020130
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076725, 0.0077156
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021632, 0.0021753
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159604, 0.0160499
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021121, 0.0021240
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119947, 0.0119278
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033325, 0.0033139
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030249, 0.0030080
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112884, 0.0112254
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087368, 0.0087858
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007580, 0.0007538

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020351, upper bound: 0.0019644
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020326, upper bound: 0.0019648
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076939, 0.0076933
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021692, 0.0021690
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160049, 0.0160036
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021180, 0.0021178
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119601, 0.0119611
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033229, 0.0033231
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030162, 0.0030164
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112558, 0.0112567
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087611, 0.0087604
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007558, 0.0007559

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020003, upper bound: 0.0019942
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019956, upper bound: 0.0019958
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077142, 0.0076731
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021749, 0.0021633
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160471, 0.0159616
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021236, 0.0021123
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119287, 0.0119926
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033142, 0.0033319
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030083, 0.0030244
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112263, 0.0112864
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087842, 0.0087374
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007538, 0.0007579

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020225, upper bound: 0.0019778
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020210, upper bound: 0.0019806
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077364, 0.0076513
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021812, 0.0021572
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160932, 0.0159162
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021297, 0.0021063
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118948, 0.0120270
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033047, 0.0033415
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029997, 0.0030330
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111943, 0.0113188
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088094, 0.0087126
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007517, 0.0007600

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019875, upper bound: 0.0020091
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0020100
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076720, 0.0077155
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021630, 0.0021753
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159593, 0.0160498
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021120, 0.0021239
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119946, 0.0119270
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033325, 0.0033137
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030249, 0.0030078
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112883, 0.0112246
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087361, 0.0087857
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007580, 0.0007537

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020333, upper bound: 0.0019698
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020307, upper bound: 0.0019706
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076939, 0.0076933
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021692, 0.0021690
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160049, 0.0160036
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021180, 0.0021178
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119601, 0.0119611
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033229, 0.0033231
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030162, 0.0030164
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112558, 0.0112567
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087611, 0.0087604
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007558, 0.0007559

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019978, upper bound: 0.0019982
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019931, upper bound: 0.0019995
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077143, 0.0076732
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021750, 0.0021634
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160474, 0.0159619
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021236, 0.0021123
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119289, 0.0119928
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033142, 0.0033320
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030083, 0.0030244
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112264, 0.0112866
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087844, 0.0087375
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007538, 0.0007579

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020205, upper bound: 0.0019818
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020188, upper bound: 0.0019851
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077366, 0.0076519
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021812, 0.0021574
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160938, 0.0159176
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021298, 0.0021064
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118958, 0.0120275
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033050, 0.0033416
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029999, 0.0030332
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111953, 0.0113192
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088098, 0.0087133
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007517, 0.0007601

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019848, upper bound: 0.0020121
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0020130
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076519, 0.0077319
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021574, 0.0021799
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159176, 0.0160840
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021064, 0.0021285
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120202, 0.0118958
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033396, 0.0033050
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030313, 0.0029999
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113123, 0.0111953
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087133, 0.0088044
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007596, 0.0007517

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020130, upper bound: 0.0019834
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020121, upper bound: 0.0019847
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076732, 0.0077095
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021634, 0.0021736
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159619, 0.0160373
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021123, 0.0021223
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119853, 0.0119289
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033299, 0.0033142
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030225, 0.0030083
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112795, 0.0112264
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087375, 0.0087788
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007574, 0.0007538

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019851, upper bound: 0.0020188
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019818, upper bound: 0.0020205
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076933, 0.0076898
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021690, 0.0021680
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160036, 0.0159963
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021178, 0.0021169
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119547, 0.0119601
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033214, 0.0033229
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030148, 0.0030162
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112507, 0.0112558
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087604, 0.0087564
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007555, 0.0007558

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019995, upper bound: 0.0019931
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0019979
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077155, 0.0076665
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021753, 0.0021615
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160498, 0.0159480
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021239, 0.0021105
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119185, 0.0119946
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033113, 0.0033325
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030057, 0.0030249
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112166, 0.0112883
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087857, 0.0087299
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007532, 0.0007580

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019706, upper bound: 0.0020307
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019698, upper bound: 0.0020333
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076513, 0.0077316
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021572, 0.0021798
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159162, 0.0160834
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021063, 0.0021284
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120197, 0.0118948
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033394, 0.0033047
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030312, 0.0029997
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113119, 0.0111943
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087126, 0.0088041
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007596, 0.0007517

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020100, upper bound: 0.0019859
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020091, upper bound: 0.0019875
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076731, 0.0077093
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021633, 0.0021735
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159616, 0.0160369
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021123, 0.0021222
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119849, 0.0119287
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033298, 0.0033142
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030224, 0.0030083
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112792, 0.0112263
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087374, 0.0087786
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007574, 0.0007538

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019806, upper bound: 0.0020210
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019778, upper bound: 0.0020225
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076933, 0.0076896
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021690, 0.0021680
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160036, 0.0159959
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021178, 0.0021168
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119543, 0.0119601
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033213, 0.0033229
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030147, 0.0030162
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112503, 0.0112558
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087604, 0.0087562
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007554, 0.0007558

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019956
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019942, upper bound: 0.0020003
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077156, 0.0076667
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021753, 0.0021615
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160499, 0.0159484
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021240, 0.0021105
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119188, 0.0119947
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033114, 0.0033325
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030058, 0.0030249
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112169, 0.0112884
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087858, 0.0087302
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007532, 0.0007580

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019648, upper bound: 0.0020327
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019644, upper bound: 0.0020350
time: 1.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076776, 0.0077109
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021646, 0.0021740
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159710, 0.0160403
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021135, 0.0021227
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119875, 0.0119358
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033305, 0.0033161
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030231, 0.0030100
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112816, 0.0112329
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087426, 0.0087805
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007575, 0.0007543

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020130, upper bound: 0.0019834
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020121, upper bound: 0.0019847
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076989, 0.0076886
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021706, 0.0021677
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160153, 0.0159939
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021194, 0.0021165
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119529, 0.0119689
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033209, 0.0033253
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030143, 0.0030184
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112490, 0.0112640
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087668, 0.0087551
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007553, 0.0007564

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019851, upper bound: 0.0020188
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019818, upper bound: 0.0020205
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077190, 0.0076682
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021763, 0.0021620
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160570, 0.0159514
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021249, 0.0021109
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119211, 0.0120000
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033120, 0.0033340
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030063, 0.0030262
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112191, 0.0112934
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087896, 0.0087318
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007533, 0.0007583

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019995, upper bound: 0.0019931
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0019979
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077412, 0.0076463
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021825, 0.0021558
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161032, 0.0159058
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021310, 0.0021049
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118870, 0.0120346
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033026, 0.0033436
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029977, 0.0030349
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111870, 0.0113259
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088149, 0.0087069
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007512, 0.0007605

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019706, upper bound: 0.0020307
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019698, upper bound: 0.0020332
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076770, 0.0077106
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021644, 0.0021739
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159697, 0.0160397
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021133, 0.0021226
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119871, 0.0119347
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033304, 0.0033158
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030230, 0.0030098
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112812, 0.0112319
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087418, 0.0087802
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007575, 0.0007542

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020100, upper bound: 0.0019859
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020091, upper bound: 0.0019875
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076988, 0.0076885
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021706, 0.0021677
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160151, 0.0159937
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021193, 0.0021165
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119527, 0.0119687
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033208, 0.0033253
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030143, 0.0030183
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112488, 0.0112639
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087667, 0.0087550
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007553, 0.0007563

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019806, upper bound: 0.0020210
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019778, upper bound: 0.0020225
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077190, 0.0076682
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021763, 0.0021620
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160571, 0.0159514
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021249, 0.0021109
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119211, 0.0120001
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033120, 0.0033340
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030063, 0.0030262
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112191, 0.0112934
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087897, 0.0087318
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007533, 0.0007583

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019956
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019942, upper bound: 0.0020003
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077413, 0.0076468
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021826, 0.0021559
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161034, 0.0159070
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021310, 0.0021050
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118879, 0.0120347
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033028, 0.0033436
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029979, 0.0030350
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111878, 0.0113260
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088150, 0.0087075
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007512, 0.0007605

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019648, upper bound: 0.0020326
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019644, upper bound: 0.0020351
time: 1.94 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.95 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020351, upper bound: 0.0019644
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020326, upper bound: 0.0019648
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020003, upper bound: 0.0019942
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019956, upper bound: 0.0019958
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020225, upper bound: 0.0019778
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020210, upper bound: 0.0019806
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019875, upper bound: 0.0020091
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0020100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020333, upper bound: 0.0019698
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020307, upper bound: 0.0019706
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019978, upper bound: 0.0019982
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019931, upper bound: 0.0019995
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020205, upper bound: 0.0019818
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020188, upper bound: 0.0019851
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019848, upper bound: 0.0020121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0020130
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020351, upper bound: 0.0019644
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020326, upper bound: 0.0019648
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020003, upper bound: 0.0019942
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019956, upper bound: 0.0019958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020225, upper bound: 0.0019778
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020210, upper bound: 0.0019806
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019875, upper bound: 0.0020091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0020100
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020333, upper bound: 0.0019698
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020307, upper bound: 0.0019706
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019978, upper bound: 0.0019982
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019931, upper bound: 0.0019995
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020205, upper bound: 0.0019818
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020188, upper bound: 0.0019851
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019848, upper bound: 0.0020121
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0020130
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020130, upper bound: 0.0019834
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020121, upper bound: 0.0019847
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019851, upper bound: 0.0020188
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019818, upper bound: 0.0020205
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019995, upper bound: 0.0019931
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0019979
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019706, upper bound: 0.0020307
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019698, upper bound: 0.0020333
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020100, upper bound: 0.0019859
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020091, upper bound: 0.0019875
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019806, upper bound: 0.0020210
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019778, upper bound: 0.0020225
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019956
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019942, upper bound: 0.0020003
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019648, upper bound: 0.0020327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019644, upper bound: 0.0020350
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020130, upper bound: 0.0019834
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020121, upper bound: 0.0019847
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019851, upper bound: 0.0020188
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019818, upper bound: 0.0020205
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019995, upper bound: 0.0019931
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0019979
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019706, upper bound: 0.0020307
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019698, upper bound: 0.0020332
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020100, upper bound: 0.0019859
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0020091, upper bound: 0.0019875
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019806, upper bound: 0.0020210
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019778, upper bound: 0.0020225
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019956
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019942, upper bound: 0.0020003
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019648, upper bound: 0.0020326
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 5, lower bound: -0.0019644, upper bound: 0.0020351

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075695, 0.0077264
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021341, 0.0021783
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157462, 0.0160724
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020838, 0.0021269
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120115, 0.0117677
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033372, 0.0032694
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030291, 0.0029676
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113042, 0.0110747
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086195, 0.0087980
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007591, 0.0007436

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020010, upper bound: 0.0019234
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0019295
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076369, 0.0076594
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021531, 0.0021595
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158863, 0.0159331
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021023, 0.0021085
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119074, 0.0118724
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033082, 0.0032985
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030029, 0.0029940
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112062, 0.0111733
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086962, 0.0087218
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007525, 0.0007503

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019238
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019798, upper bound: 0.0019298
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075909, 0.0077045
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021402, 0.0021722
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157906, 0.0160269
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020896, 0.0021209
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119775, 0.0118009
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033277, 0.0032787
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030205, 0.0029760
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112721, 0.0111060
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086438, 0.0087731
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007569, 0.0007457

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019653, upper bound: 0.0019520
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019517, upper bound: 0.0019600
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076578, 0.0076369
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021590, 0.0021531
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159298, 0.0158863
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021081, 0.0021023
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118724, 0.0119049
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032985, 0.0033075
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029940, 0.0030023
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111733, 0.0112039
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087200, 0.0086962
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007503, 0.0007523

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019619, upper bound: 0.0019536
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019463, upper bound: 0.0019610
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076112, 0.0076815
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021459, 0.0021657
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158329, 0.0159791
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020952, 0.0021146
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119418, 0.0118325
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033178, 0.0032874
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030115, 0.0029840
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112386, 0.0111357
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086669, 0.0087470
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007546, 0.0007477

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019889, upper bound: 0.0019350
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019728, upper bound: 0.0019434
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076814, 0.0076174
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021657, 0.0021476
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159789, 0.0158457
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021146, 0.0020969
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118421, 0.0119417
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032901, 0.0033177
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029864, 0.0030115
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111447, 0.0112384
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087469, 0.0086740
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007483, 0.0007546

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019880, upper bound: 0.0019378
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019702, upper bound: 0.0019453
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076334, 0.0076595
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021521, 0.0021595
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158789, 0.0159333
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021013, 0.0021085
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119075, 0.0118669
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033083, 0.0032970
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030029, 0.0029927
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112063, 0.0111681
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086921, 0.0087219
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007525, 0.0007499

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0019636
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019413, upper bound: 0.0019747
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077014, 0.0075943
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021713, 0.0021411
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160206, 0.0157978
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021201, 0.0020906
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118063, 0.0119728
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032801, 0.0033264
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029774, 0.0030194
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111110, 0.0112677
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087697, 0.0086477
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007461, 0.0007566

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019519, upper bound: 0.0019662
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019395, upper bound: 0.0019758
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075690, 0.0077270
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021340, 0.0021785
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157451, 0.0160737
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020836, 0.0021271
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120125, 0.0117669
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033374, 0.0032692
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030294, 0.0029674
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113051, 0.0110739
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086189, 0.0087988
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007591, 0.0007436

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019287
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019808, upper bound: 0.0019348
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076360, 0.0076592
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021529, 0.0021594
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158845, 0.0159326
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021021, 0.0021084
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119071, 0.0118711
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033081, 0.0032981
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030028, 0.0029937
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112059, 0.0111720
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086952, 0.0087216
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007525, 0.0007502

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019969, upper bound: 0.0019293
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019769, upper bound: 0.0019354
time: 1.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075909, 0.0077052
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021402, 0.0021724
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157906, 0.0160283
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020896, 0.0021211
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119786, 0.0118009
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033280, 0.0032787
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030208, 0.0029760
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112732, 0.0111060
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086438, 0.0087739
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007570, 0.0007457

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019629, upper bound: 0.0019564
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019637
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076569, 0.0076367
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021588, 0.0021531
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159280, 0.0158858
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021078, 0.0021022
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118721, 0.0119036
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032984, 0.0033072
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029940, 0.0030019
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111729, 0.0112026
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087190, 0.0086959
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007502, 0.0007522

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019593, upper bound: 0.0019576
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019440, upper bound: 0.0019647
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076114, 0.0076824
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021459, 0.0021660
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158332, 0.0159809
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020953, 0.0021148
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119432, 0.0118327
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033182, 0.0032875
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030119, 0.0029840
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112398, 0.0111359
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086671, 0.0087480
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007547, 0.0007478

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019870, upper bound: 0.0019387
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019695, upper bound: 0.0019473
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076808, 0.0076173
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021655, 0.0021476
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159777, 0.0158456
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021144, 0.0020969
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118420, 0.0119407
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032901, 0.0033175
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029864, 0.0030113
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111446, 0.0112376
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087462, 0.0086739
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007483, 0.0007546

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019858, upper bound: 0.0019424
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019670, upper bound: 0.0019496
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076337, 0.0076606
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021522, 0.0021598
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158796, 0.0159356
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021014, 0.0021088
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119093, 0.0118674
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033087, 0.0032971
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030033, 0.0029928
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112079, 0.0111685
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086925, 0.0087232
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007526, 0.0007499

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019506, upper bound: 0.0019673
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0019777
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077011, 0.0075945
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021712, 0.0021412
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160198, 0.0157981
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021200, 0.0020906
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118065, 0.0119722
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032802, 0.0033262
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029774, 0.0030192
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111113, 0.0112672
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087692, 0.0086479
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007461, 0.0007566

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019699
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019361, upper bound: 0.0019786
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075958, 0.0077058
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021415, 0.0021725
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158008, 0.0160295
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020910, 0.0021213
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119795, 0.0118085
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033283, 0.0032808
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030211, 0.0029779
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112740, 0.0111132
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086494, 0.0087746
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007570, 0.0007462

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020010, upper bound: 0.0019234
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0019295
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076632, 0.0076383
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021605, 0.0021535
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159409, 0.0158892
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021095, 0.0021027
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118746, 0.0119132
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032991, 0.0033099
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029946, 0.0030043
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111753, 0.0112117
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087261, 0.0086977
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007504, 0.0007528

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019238
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019798, upper bound: 0.0019298
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076172, 0.0076856
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021476, 0.0021668
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158453, 0.0159875
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020969, 0.0021157
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119481, 0.0118418
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033195, 0.0032900
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030131, 0.0029863
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112445, 0.0111444
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086737, 0.0087516
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007550, 0.0007483

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019653, upper bound: 0.0019520
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019517, upper bound: 0.0019600
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076841, 0.0076160
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021664, 0.0021472
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159844, 0.0158428
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021153, 0.0020965
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118399, 0.0119458
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032895, 0.0033189
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029859, 0.0030125
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111427, 0.0112423
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087499, 0.0086724
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007482, 0.0007549

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019619, upper bound: 0.0019536
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019463, upper bound: 0.0019610
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076375, 0.0076620
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021533, 0.0021602
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158875, 0.0159384
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021025, 0.0021092
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119114, 0.0118733
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033093, 0.0032988
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030039, 0.0029943
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112099, 0.0111741
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086969, 0.0087247
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007527, 0.0007503

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019889, upper bound: 0.0019350
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019728, upper bound: 0.0019434
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077077, 0.0075958
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021731, 0.0021415
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160336, 0.0158008
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021218, 0.0020910
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118086, 0.0119825
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032808, 0.0033291
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029779, 0.0030218
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111132, 0.0112769
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087768, 0.0086494
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007462, 0.0007572

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019880, upper bound: 0.0019378
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019702, upper bound: 0.0019453
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076596, 0.0076411
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021595, 0.0021543
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159336, 0.0158951
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021086, 0.0021035
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118790, 0.0119078
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033003, 0.0033083
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029957, 0.0030030
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111794, 0.0112065
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087221, 0.0087010
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007507, 0.0007525

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0019636
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019413, upper bound: 0.0019747
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077277, 0.0075740
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021787, 0.0021354
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160752, 0.0157554
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021273, 0.0020850
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117746, 0.0120136
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032713, 0.0033377
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029694, 0.0030297
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110812, 0.0113061
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087996, 0.0086245
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007441, 0.0007592

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019519, upper bound: 0.0019662
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019395, upper bound: 0.0019757
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075953, 0.0077062
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021414, 0.0021727
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157997, 0.0160305
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020908, 0.0021214
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119802, 0.0118077
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033284, 0.0032805
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030212, 0.0029777
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112747, 0.0111124
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086488, 0.0087751
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007571, 0.0007462

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019287
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019808, upper bound: 0.0019349
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076623, 0.0076382
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021603, 0.0021535
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159392, 0.0158890
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021093, 0.0021027
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118744, 0.0119119
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032991, 0.0033095
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029946, 0.0030040
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111752, 0.0112105
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087251, 0.0086977
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007504, 0.0007528

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019969, upper bound: 0.0019293
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019769, upper bound: 0.0019354
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076172, 0.0076861
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021476, 0.0021670
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158453, 0.0159887
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020969, 0.0021158
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119489, 0.0118418
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033198, 0.0032900
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030133, 0.0029863
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112453, 0.0111444
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086737, 0.0087522
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007551, 0.0007483

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019629, upper bound: 0.0019563
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019637
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076832, 0.0076160
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021662, 0.0021472
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159826, 0.0158428
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021150, 0.0020965
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118399, 0.0119444
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032895, 0.0033185
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029859, 0.0030122
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111427, 0.0112410
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087489, 0.0086724
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007482, 0.0007548

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019593, upper bound: 0.0019576
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019440, upper bound: 0.0019647
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076376, 0.0076631
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021533, 0.0021605
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158878, 0.0159408
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021025, 0.0021095
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119131, 0.0118736
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033098, 0.0032988
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030043, 0.0029943
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112116, 0.0111743
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086970, 0.0087260
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007528, 0.0007503

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019870, upper bound: 0.0019387
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019695, upper bound: 0.0019472
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077071, 0.0075959
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021729, 0.0021416
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160324, 0.0158011
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021216, 0.0020910
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118087, 0.0119816
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032808, 0.0033288
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029780, 0.0030216
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111133, 0.0112760
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087761, 0.0086495
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007462, 0.0007572

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019858, upper bound: 0.0019424
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019670, upper bound: 0.0019496
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076599, 0.0076420
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021596, 0.0021546
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159342, 0.0158968
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021086, 0.0021037
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118803, 0.0119082
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033007, 0.0033085
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029960, 0.0030031
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111807, 0.0112070
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087224, 0.0087019
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007508, 0.0007525

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019506, upper bound: 0.0019673
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0019777
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077273, 0.0075746
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021786, 0.0021356
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160744, 0.0157568
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021272, 0.0020852
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117756, 0.0120130
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032716, 0.0033376
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029696, 0.0030295
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110822, 0.0113056
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087992, 0.0086253
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007441, 0.0007591

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019699
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019361, upper bound: 0.0019786
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075746, 0.0077213
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021356, 0.0021769
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157568, 0.0160619
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020852, 0.0021255
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120037, 0.0117756
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033350, 0.0032716
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030272, 0.0029696
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112968, 0.0110822
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086253, 0.0087923
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007586, 0.0007441

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019786, upper bound: 0.0019361
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019699, upper bound: 0.0019495
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076420, 0.0076546
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021546, 0.0021581
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158968, 0.0159232
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021037, 0.0021072
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119000, 0.0118803
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033062, 0.0033007
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030010, 0.0029960
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111992, 0.0111807
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087019, 0.0087164
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007520, 0.0007508

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019777, upper bound: 0.0019378
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019673, upper bound: 0.0019506
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075959, 0.0076996
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021416, 0.0021708
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158011, 0.0160167
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020910, 0.0021195
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119698, 0.0118087
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033256, 0.0032808
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030186, 0.0029780
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112650, 0.0111133
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086495, 0.0087675
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007564, 0.0007462

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0019670
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019424, upper bound: 0.0019858
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076631, 0.0076322
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021605, 0.0021518
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159408, 0.0158765
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021095, 0.0021010
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118651, 0.0119131
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032965, 0.0033098
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029922, 0.0030043
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111664, 0.0112116
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087260, 0.0086908
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007498, 0.0007528

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019473, upper bound: 0.0019695
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019387, upper bound: 0.0019870
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076160, 0.0076766
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021472, 0.0021643
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158428, 0.0159689
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020965, 0.0021132
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119342, 0.0118399
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033157, 0.0032895
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030096, 0.0029859
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112314, 0.0111427
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086724, 0.0087414
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007542, 0.0007482

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019647, upper bound: 0.0019440
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019576, upper bound: 0.0019593
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076861, 0.0076125
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021670, 0.0021462
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159887, 0.0158355
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021158, 0.0020956
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118345, 0.0119489
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032880, 0.0033198
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029845, 0.0030133
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111376, 0.0112453
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087522, 0.0086684
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007479, 0.0007551

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019638, upper bound: 0.0019495
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019564, upper bound: 0.0019629
time: 1.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076382, 0.0076544
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021535, 0.0021580
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158890, 0.0159226
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021027, 0.0021071
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118996, 0.0118744
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033061, 0.0032991
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030009, 0.0029946
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111988, 0.0111752
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086977, 0.0087161
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007520, 0.0007504

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019353, upper bound: 0.0019769
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019293, upper bound: 0.0019969
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077062, 0.0075892
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021727, 0.0021397
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160305, 0.0157872
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021214, 0.0020892
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117983, 0.0119802
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032779, 0.0033284
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029754, 0.0030212
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111036, 0.0112747
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087751, 0.0086419
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007456, 0.0007571

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019349, upper bound: 0.0019808
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019287, upper bound: 0.0019990
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075740, 0.0077218
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021354, 0.0021771
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157554, 0.0160629
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020850, 0.0021257
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120044, 0.0117746
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033352, 0.0032713
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030273, 0.0029694
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112975, 0.0110812
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086245, 0.0087928
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007586, 0.0007441

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019757, upper bound: 0.0019395
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019662, upper bound: 0.0019519
time: 1.73 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 6.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0020010, upper bound: 0.0019234
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0019295
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019238
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019798, upper bound: 0.0019298
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019653, upper bound: 0.0019520
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019517, upper bound: 0.0019600
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019619, upper bound: 0.0019536
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019463, upper bound: 0.0019610
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019889, upper bound: 0.0019350
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019728, upper bound: 0.0019434
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019880, upper bound: 0.0019378
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019702, upper bound: 0.0019453
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0019636
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019413, upper bound: 0.0019747
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019519, upper bound: 0.0019662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019395, upper bound: 0.0019758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019287
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019808, upper bound: 0.0019348
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019969, upper bound: 0.0019293
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019769, upper bound: 0.0019354
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019629, upper bound: 0.0019564
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019637
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019593, upper bound: 0.0019576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019440, upper bound: 0.0019647
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019870, upper bound: 0.0019387
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019695, upper bound: 0.0019473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019858, upper bound: 0.0019424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019670, upper bound: 0.0019496
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019506, upper bound: 0.0019673
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0019777
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019699
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019361, upper bound: 0.0019786
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0020010, upper bound: 0.0019234
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019834, upper bound: 0.0019295
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019238
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019798, upper bound: 0.0019298
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019653, upper bound: 0.0019520
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019517, upper bound: 0.0019600
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019619, upper bound: 0.0019536
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019463, upper bound: 0.0019610
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019889, upper bound: 0.0019350
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019728, upper bound: 0.0019434
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019880, upper bound: 0.0019378
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019702, upper bound: 0.0019453
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019534, upper bound: 0.0019636
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019413, upper bound: 0.0019747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019519, upper bound: 0.0019662
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019395, upper bound: 0.0019757
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019990, upper bound: 0.0019287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019808, upper bound: 0.0019349
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019969, upper bound: 0.0019293
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019769, upper bound: 0.0019354
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019629, upper bound: 0.0019563
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019637
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019593, upper bound: 0.0019576
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019440, upper bound: 0.0019647
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019870, upper bound: 0.0019387
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019695, upper bound: 0.0019472
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019858, upper bound: 0.0019424
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019670, upper bound: 0.0019496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019506, upper bound: 0.0019673
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019378, upper bound: 0.0019777
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019495, upper bound: 0.0019699
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019361, upper bound: 0.0019786
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019786, upper bound: 0.0019361
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019699, upper bound: 0.0019495
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019777, upper bound: 0.0019378
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019673, upper bound: 0.0019506
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0019670
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019424, upper bound: 0.0019858
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019473, upper bound: 0.0019695
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019387, upper bound: 0.0019870
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019647, upper bound: 0.0019440
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019576, upper bound: 0.0019593
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019638, upper bound: 0.0019495
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019564, upper bound: 0.0019629
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019353, upper bound: 0.0019769
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019293, upper bound: 0.0019969
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019349, upper bound: 0.0019808
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019287, upper bound: 0.0019990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019757, upper bound: 0.0019395
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.38
Output dim: 5, lower bound: -0.0019662, upper bound: 0.0019519
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0020091, upper bound: 0.0019875
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019806, upper bound: 0.0020210
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019778, upper bound: 0.0020225
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019956
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019942, upper bound: 0.0020003
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019648, upper bound: 0.0020327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019644, upper bound: 0.0020350
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0020130, upper bound: 0.0019834
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0020121, upper bound: 0.0019847
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019851, upper bound: 0.0020188
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019818, upper bound: 0.0020205
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019995, upper bound: 0.0019931
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019982, upper bound: 0.0019979
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019706, upper bound: 0.0020307
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019698, upper bound: 0.0020332
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0020100, upper bound: 0.0019859
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0020091, upper bound: 0.0019875
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019806, upper bound: 0.0020210
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019778, upper bound: 0.0020225
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019958, upper bound: 0.0019956
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019942, upper bound: 0.0020003
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019648, upper bound: 0.0020326
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.38
Output dim: 5, lower bound: -0.0019644, upper bound: 0.0020351

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.01 + 595.31 = 600.32 seconds
